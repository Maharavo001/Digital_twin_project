import wntr
import wntr.network.controls as controls
import sqlite3
import csv
import sys
import pandas as pd
import yaml
import time
import os

class Simulation:

    def __init__(self):
        config_file_path = sys.argv[1]
        config_options = self.load_config(config_file_path)

        if "simulation_tpye" in config_options and config_options['simulation_tpye'] == "Batch":
            self.week_index = int(sys.argv[2])
        else:
            self.week_index = int(config_options['week_index'])

        self.db_path = config_options['db_path']
        self.db_name = config_options['db_name']
        self.conn = sqlite3.connect(self.db_path)
        self.c = self.conn.cursor()

        self.output_path = config_options['output_ground_truth_path']
        self.simulation_days = int(config_options['duration_days'])

        inp_file = config_options['inp_file']
        self.wn = wntr.network.WaterNetworkModel(inp_file)

        self.node_list = list(self.wn.node_name_list)
        self.link_list = list(self.wn.link_name_list)

        self.tank_list = self.get_node_list_by_type(self.node_list, 'Tank')
        self.junction_list = self.get_node_list_by_type(self.node_list, 'Junction')
        self.pump_list = self.get_link_list_by_type(self.link_list, 'Pump')
        self.valve_list = self.get_link_list_by_type(self.link_list, 'Valve')

        list_header = ["Timestamps"]
        list_header.extend(self.create_node_header(self.tank_list))
        list_header.extend(self.create_node_header(self.junction_list))
        list_header.extend(self.create_link_header(self.pump_list))
        list_header.extend(self.create_link_header(self.valve_list))
        list_header.extend(["Attack#01", "Attack#02"])

        self.results_list = []
        self.results_list.append(list_header)

        self.initialize_simulation(config_options)

        dummy_condition = controls.ValueCondition(self.wn.get_node(self.tank_list[0]), 'level', '>=', -1)

        self.control_list = []
        for valve in self.valve_list:
            self.control_list.append(self.create_control_dict(valve, dummy_condition))
        for pump in self.pump_list:
            self.control_list.append(self.create_control_dict(pump, dummy_condition))

        for control in self.control_list:
            an_action = controls.ControlAction(control['actuator'], control['parameter'], control['value'])
            a_control = controls.Control(control['condition'], an_action, name=control['name'])
            self.wn.add_control(control['name'], a_control)

        simulator_string = config_options['simulator']
        if simulator_string == 'pdd':
            print('Running simulation using PDD')
            self.wn.options.hydraulic.demand_model = 'PDD'
        elif simulator_string == 'dd':
            print('Running simulation using DD')
        else:
            print('Invalid simulation mode, exiting...')
            sys.exit(1)

        self.sim = wntr.sim.WNTRSimulator(self.wn)
        print("Starting simulation for " + str(config_options['inp_file']) + " topology ")

    def load_config(self, config_path):
        with open(config_path) as config_file:
            options = yaml.load(config_file, Loader=yaml.FullLoader)
        return options

    def initialize_simulation(self, config_options):
        if self.simulation_days == 7:
            limit = 167
        else:
            limit = 239

        if 'initial_custom_flag' in config_options:
            custom_initial_conditions_flag = bool(config_options['initial_custom_flag'])
            if custom_initial_conditions_flag:
                demand_patterns_path = config_options['demand_patterns_path']
                starting_demand_path = config_options['starting_demand_path']
                initial_tank_levels_path = config_options['initial_tank_levels_path']

                print("Running simulation with week index: " + str(self.week_index))
                total_demands = pd.read_csv(demand_patterns_path, index_col=0)
                demand_starting_points = pd.read_csv(starting_demand_path, index_col=0)
                initial_tank_levels = pd.read_csv(initial_tank_levels_path, index_col=0)
                week_start = demand_starting_points.iloc[self.week_index][0]
                week_demands = total_demands.loc[week_start:week_start + limit, :]

                for name, pat in self.wn.patterns():
                    pat.multipliers = week_demands[name].values.tolist()

                self.wn.get_node('TANK').init_level = float(initial_tank_levels.iloc[self.week_index]['TANK'])

    def get_node_list_by_type(self, a_list, a_type):
        return [str(node) for node in a_list if self.wn.get_node(node).node_type == a_type]

    def get_link_list_by_type(self, a_list, a_type):
        return [str(link) for link in a_list if self.wn.get_link(link).link_type == a_type]

    def create_node_header(self, a_list):
        return [node + "_LEVEL" for node in a_list]

    def create_link_header(self, a_list):
        result = []
        for link in a_list:
            result.append(link + "_FLOW")
            result.append(link + "_STATUS")
        return result

    def create_control_dict(self, actuator, dummy_condition):
        act_dict = dict.fromkeys(['actuator', 'parameter', 'value', 'condition', 'name'])
        act_dict['actuator'] = self.wn.get_link(actuator)
        act_dict['parameter'] = 'status'
        act_dict['condition'] = dummy_condition
        act_dict['name'] = actuator
        if type(self.wn.get_link(actuator).status) is int:
            act_dict['value'] = act_dict['actuator'].status
        else:
            act_dict['value'] = act_dict['actuator'].status.value
        return act_dict

    def register_results(self, results):
        values_list = [results.timestamp]

        for tank in self.tank_list:
            values_list.append(self.wn.get_node(tank).level)

        for junction in self.junction_list:
            values_list.append(self.wn.get_node(junction).head - self.wn.get_node(junction).elevation)

        for pump in self.pump_list:
            values_list.append(self.wn.get_link(pump).flow)
            status = self.wn.get_link(pump).status
            values_list.append(status if isinstance(status, int) else status.value)

        for valve in self.valve_list:
            values_list.append(self.wn.get_link(valve).flow)
            status = self.wn.get_link(valve).status
            values_list.append(status if isinstance(status, int) else status.value)

        rows = self.c.execute(f"SELECT value FROM {self.db_name} WHERE name = 'ATT_1' AND pid=1").fetchall()
        self.conn.commit()
        attack1 = int(rows[0][0]) if rows else 0

        rows = self.c.execute(f"SELECT value FROM {self.db_name} WHERE name = 'ATT_2' AND pid=1").fetchall()
        self.conn.commit()
        attack2 = int(rows[0][0]) if rows else 0

        values_list.extend([attack1, attack2])
        return values_list

    def update_controls(self):
        for control in self.control_list:
            self.update_control(control)

    def update_control(self, control):
        act_name = '\'' + control['name'] + '\''
        rows_1 = self.c.execute(f"SELECT value FROM {self.db_name} WHERE name = {act_name} AND pid=1").fetchall()
        self.conn.commit()
        if rows_1:
            new_status = int(rows_1[0][0])
            control['value'] = new_status

            new_action = controls.ControlAction(control['actuator'], control['parameter'], control['value'])
            new_control = controls.Control(control['condition'], new_action, name=control['name'])
            self.wn.remove_control(control['name'])
            self.wn.add_control(control['name'], new_control)

    def write_results(self, results):
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, self.output_path)
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(results)

    def main(self):
        self.wn.options.time.duration = self.wn.options.time.hydraulic_timestep
        master_time = 0

        self.simulation_days = 1
        iteration_limit = (self.simulation_days * 24 * 3600) / self.wn.options.time.hydraulic_timestep

        print(f"Simulation will run for {self.simulation_days} days. Hydraulic timestep is {self.wn.options.time.hydraulic_timestep} for a total of {iteration_limit} iterations")

        while master_time <= iteration_limit:
            rows = self.c.execute(f"SELECT value FROM {self.db_name} WHERE name = 'CONTROL' AND pid=1").fetchall()
            self.conn.commit()
            if not rows:
                print("No CONTROL value found in DB, stopping simulation.")
                break
            control = int(rows[0][0])

            if control == 1:
                self.update_controls()
                print(f"ITERATION {master_time} ------------- ", flush=True)

                results = self.sim.run_sim(convergence_error=True)
                values_list = self.register_results(results)
                self.results_list.append(values_list)

                for tank in self.tank_list:
                    level = self.wn.get_node(tank).level
                    print(f"  Tank {tank} level: {level:.2f}")

                for pump in self.pump_list:
                    status = self.wn.get_link(pump).status
                    flow = self.wn.get_link(pump).flow
                    print(f"  Pump {pump} status: {status}, flow: {flow:.2f}")

                master_time += 1

                for tank in self.tank_list:
                    tank_name = '\'' + tank + '\''
                    a_level = self.wn.get_node(tank).level
                    query = f"UPDATE {self.db_name} SET value = {a_level} WHERE name = {tank_name} AND pid=1"
                    self.c.execute(query)
                    self.conn.commit()

                self.c.execute(f"UPDATE {self.db_name} SET value = 0 WHERE name = 'CONTROL' AND pid=1")
                self.conn.commit()

                if results:
                    time.sleep(0.5)
            else:
                time.sleep(0.5)

        self.write_results(self.results_list)

if __name__ == "__main__":
    simulation = Simulation()
    simulation.main()
    exit(0)

