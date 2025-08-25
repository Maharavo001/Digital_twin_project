import warnings
import sys
import os

# Completely suppress pkg_resources warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Redirect warnings to null (cross-platform)
if sys.platform.startswith('win'):
    from tempfile import TemporaryFile
    null_file = TemporaryFile()
else:
    null_file = open(os.devnull, 'w')

sys.stderr = null_file

import wntr  # Import wntr after suppressing warnings
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

class Simulation:
    def __init__(self, inp_file: str):
        """Initialize with just the INP file path"""
        try:
            # Load water network model
            self.wn = wntr.network.WaterNetworkModel(inp_file)
            
            # Initialize components
            self._setup_network_components()
            
            # Fast simulator setup
            self.sim = wntr.sim.WNTRSimulator(self.wn)
            self.wn.options.hydraulic.demand_model = 'pdd'
            
        except Exception as e:
            raise ValueError(f"Failed to initialize network: {str(e)}")

    def _setup_network_components(self):
        """Shared network setup"""
        self.node_list = list(self.wn.node_name_list)
        self.link_list = list(self.wn.link_name_list)
        self.tank_list = [n for n in self.node_list if self.wn.get_node(n).node_type == 'Tank']
        self.junction_list = [n for n in self.node_list if self.wn.get_node(n).node_type == 'Junction']
        self.pump_list = [l for l in self.link_list if self.wn.get_link(l).link_type == 'Pump']
        self.valve_list = [l for l in self.link_list if self.wn.get_link(l).link_type == 'Valve']

    def set_demands(self, demand_dict):
        """Update junction demands from dictionary {name: value}"""
        for junc, demand in demand_dict.items():
            if junc in self.junction_list:
                # Get the junction object
                junction = self.wn.get_node(junc)
                # Create new demand pattern if needed
                if len(junction.demand_timeseries_list) == 0:
                    junction.add_demand(base=float(demand), pattern_name=None)
                else:
                    # Modify the existing base demand
                    junction.demand_timeseries_list[0].base_value = float(demand)

    def run_step(self, demand_dict):
        """Run a single step simulation with the given demands and return results."""
        self.set_demands(demand_dict)
        self.wn.options.time.duration = self.wn.options.time.hydraulic_timestep
        results = self.sim.run_sim()
        return self.extract_results(results)

    def extract_results(self, results):
        """Extract simulation results into a dictionary."""
        return {
            "tanks": {tank: results.node["pressure"].loc[:, tank].iloc[-1] 
                     for tank in self.tank_list},
            "junctions": {junc: results.node["pressure"].loc[:, junc].iloc[-1] 
                         for junc in self.junction_list},
            "pumps": {pump: {
                "flow": results.link["flowrate"].loc[:, pump].iloc[-1],
                "status": results.link["status"].loc[:, pump].iloc[-1]
            } for pump in self.pump_list},
            "valves": {valve: {
                "flow": results.link["flowrate"].loc[:, valve].iloc[-1],
                "status": results.link["status"].loc[:, valve].iloc[-1]
            } for valve in self.valve_list}
        }

if __name__ == "__main__":
    try:
        if len(sys.argv) < 3:
            print(json.dumps({"error": "Usage: python physical_process.py <inp_file|state_json> <demand_json>"}))
            sys.exit(1)

        input_arg = sys.argv[1]
        demand_json = sys.argv[2]

        # Case 1: start from inp file
        if input_arg.endswith(".inp"):
            sim = Simulation(input_arg)
            prev_state = {"inp_file": input_arg, "demands": {}}

        # Case 2: start from state dict (json string)
        else:
            prev_state = json.loads(input_arg)
            sim = Simulation(prev_state["inp_file"])
            sim.set_demands(prev_state.get("demands", {}))

        # Merge new demands
        demand_dict = json.loads(demand_json)
        full_demand = {j: demand_dict.get(j, sim.wn.get_node(j).base_demand)
                      for j in sim.junction_list}

        # Run one step
        results = sim.run_step(full_demand)

        # New state
        new_state = {
            "inp_file": prev_state["inp_file"],
            "demands": full_demand
        }

        # Return results + new state
        print(json.dumps({"results": results, "state": new_state}, cls=NumpyEncoder))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

