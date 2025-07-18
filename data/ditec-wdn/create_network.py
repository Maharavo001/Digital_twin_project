import pandas as pd
import networkx as nx



################################# pipes ########################################

pipe = pd.read_csv('data\ditec-wdn\pipe_connectivity.csv')

Length= pd.read_parquet('data\ditec-wdn\pipe_length-0-static_input.parquet')	
Diameter =pd.read_parquet('data\ditec-wdn\pipe_diameter-0-static_input.parquet')
Roughness= pd.read_parquet('data\ditec-wdn\pipe_roughness-0-static_input.parquet')	
MinorLoss= pd.read_parquet('data\ditec-wdn\pipe_minor_loss-0-static_input.parquet')	
Status = pd.read_parquet('data\ditec-wdn\pipe_initial_status-0-static_input.parquet')	 #(Open/Closed)

# adding a map function to define the status open or close of the pipe
def map_status(val):
    return "Closed" if val == 0.0 else "Open"


#keeping only the first row


Length = Length.iloc[0].T.reset_index()
Length.columns = ['pipe_id','Length']

Diameter =Diameter.iloc[0].T.reset_index()
Diameter.columns = ['pipe_id','Diameter']

Roughness= Roughness.iloc[0].T.reset_index()
Roughness.columns = ['pipe_id','Roughness']

MinorLoss= MinorLoss.iloc[0].T.reset_index()
MinorLoss.columns = ['pipe_id','MinorLoss']

Status =Status.iloc[0].T.reset_index()
Status.columns = ['pipe_id','Status']
Status['mapped_status'] = Status['Status'].apply(map_status)



dfs = [pipe, Length, Diameter, Roughness, MinorLoss, Status]  # list of all your DataFrames
df_merged = pipe.merge(Length, on='pipe_id', how='left')
df_merged =  df_merged.merge(Diameter, on='pipe_id', how='left')
df_merged =  df_merged.merge(Roughness, on='pipe_id', how='left')
df_merged =  df_merged.merge(MinorLoss, on='pipe_id', how='left')
df_merged =  df_merged.merge(Status, on='pipe_id', how='left')


# Remove unwanted rows if any
df_merged = df_merged[df_merged['pipe_id'] != 'scenario_id']
df_merged['start_node'] = df_merged['start_node'].str.strip().replace({'J-13a': 'J-13'})
df_merged['end_node'] = df_merged['end_node'].str.strip()


pipe_lines = []
pipe_lines.append("[PIPES]")
pipe_lines.append(";ID StartNode EndNode Length Diameter Roughness MinorLoss Status")

for idx, row in df_merged.iterrows():
    line = (f"{row['pipe_id'].replace('~@', '')} {row['start_node']} {row['end_node']} "
            f"{row['Length']:.2f} {row['Diameter']:.2f} {row['Roughness']:.2f} "
            f"{row['MinorLoss']:.2f} {row['mapped_status']}")
    pipe_lines.append(line)

pipe_section = "\n".join(pipe_lines)

##################### junctions #########################

junction = pd.read_parquet('data\ditec-wdn\junction_elevation-0-static_input.parquet')
demand =pd.read_parquet('data\ditec-wdn\junction_base_demand-0-dynamic_input.parquet')

junction = junction.iloc[0].T.reset_index()
junction.columns = ['junction_id','elevation']

demand = demand.iloc[0].T.reset_index()
demand.columns = ['junction_id','base_demand']

full_junction = junction.merge(demand, on='junction_id', how='left')
# Remove any rows where 'junction_id' equals 'scenario_id'
full_junction = full_junction[full_junction['junction_id'] != 'scenario_id']

full_junction['pattern_id'] = "D_" + full_junction['junction_id'].astype(str)


junction_lines = []
junction_lines.append("[JUNCTIONS]")
junction_lines.append(";ID Elevation Demand Pattern")

for idx, row in full_junction.iterrows():
    line = f"{row['junction_id']} {row['elevation']:.2f} {row['base_demand']:.6f} {row['pattern_id']}"
    junction_lines.append(line)

junction_section = "\n".join(junction_lines)

################## RESERVOIRS #############################
reservoirs = pd.read_parquet(r'data\ditec-wdn\reservoir_base_head-0-static_input.parquet')
reservoirs = reservoirs.iloc[0].T.reset_index()
reservoirs.columns = ['R_ID','Head']
reservoirs = reservoirs[reservoirs['R_ID'] != 'scenario_id']

reservoir_lines = []
reservoir_lines.append("[RESERVOIRS]")
reservoir_lines.append(";ID Head")

for idx, row in reservoirs.iterrows():
    line = f"{row['R_ID']} {row['Head']:.2f}"
    reservoir_lines.append(line)

reservoir_section = "\n".join(reservoir_lines)

###################### Tanks #############################
tank_diameter =pd.read_parquet(r'data\ditec-wdn\tank_diameter-0-static_input.parquet')
tank_elevation =pd.read_parquet(r'data\ditec-wdn\tank_elevation-0-static_input.parquet')
tank_init_level =pd.read_parquet(r'data\ditec-wdn\tank_init_level-0-static_input.parquet')
tank_min_vol =pd.read_parquet(r'data\ditec-wdn\tank_min_vol-0-static_input.parquet')


tank_diameter = tank_diameter.iloc[0].T.reset_index()
tank_diameter.columns = ['Tank_ID','diameter']
tank_diameter = tank_diameter[tank_diameter['Tank_ID'] != 'scenario_id']

tank_elevation = tank_elevation.iloc[0].T.reset_index()
tank_elevation.columns = ['Tank_ID','elevation']
tank_elevation = tank_elevation[tank_elevation['Tank_ID'] != 'scenario_id']

tank_init_level = tank_init_level.iloc[0].T.reset_index()
tank_init_level.columns = ['Tank_ID','init_level']
tank_init_level = tank_init_level[tank_init_level['Tank_ID'] != 'scenario_id']

tank_min_vol = tank_min_vol.iloc[0].T.reset_index()
tank_min_vol.columns = ['Tank_ID','min_vol']
tank_min_vol = tank_min_vol[tank_min_vol['Tank_ID'] != 'scenario_id']

full_tank = tank_diameter.merge(tank_elevation, on='Tank_ID', how='left')
full_tank = full_tank.merge(tank_init_level, on='Tank_ID', how='left')
full_tank = full_tank.merge(tank_min_vol, on='Tank_ID', how='left')

tank_lines = []
tank_lines.append("[Tanks]")
tank_lines.append(";ID Elevation  InitLevel  MinLevel  MaxLevel  Diameter  MinVol  VolCurve")

for i, row in full_tank.iterrows():
    tank_id = row["Tank_ID"]
    elevation = row["elevation"]
    init_level = row["init_level"]
    min_level = 0  # placeholder
    max_level = init_level + 10  # arbitrary
    diameter = row["diameter"]
    min_vol = row["min_vol"]

    line = f"{tank_id} {elevation:.2f} {init_level:.2f} {min_level:.2f} {max_level:.2f} {diameter:.2f} {min_vol:.2f}"
    tank_lines.append(line)
tank_section = "\n".join(tank_lines)

######################### Pumps ############################
base_speed = pd.read_parquet(r'data\ditec-wdn\powerpump_base_speed-0-static_input.parquet')
init_status = pd.read_parquet(r'data\ditec-wdn\powerpump_initial_status-0-static_input.parquet')

base_speed = base_speed.iloc[0].T.reset_index()
base_speed.columns = ['pump_ID','base_speed']
base_speed = base_speed[base_speed['pump_ID'] != 'scenario_id']

init_status = init_status.iloc[0].T.reset_index()
init_status.columns = ['pump_ID','init_status']
init_status = init_status[init_status['pump_ID'] != 'scenario_id']

# Merge base speed and initial status on pump_ID
pump_status = pd.merge(base_speed, init_status, on='pump_ID')

# Merge with pipe connectivity to get start and end nodes
pump_data = pd.merge(pump_status, pipe, left_on='pump_ID', right_on='pipe_id')


pumps_lines = []
pumps_lines.append("[PUMPS]")
pumps_lines.append(";ID  Node1  Node2  Pump Curve  Speed  Status")

for i, row in pump_data.iterrows():
    pump_id = row['pump_ID'].replace('~@', '')
    start_node = row['start_node']
    end_node = row['end_node']
    speed = row['base_speed'] * row['init_status']  # effective speed
    state = 'OPEN' if speed > 0 else 'CLOSED'
    pump_curve = 'HEAD'  # default pump curve

    line = f"{pump_id} {start_node} {end_node} {pump_curve} {speed:.3f} {state}"
    pumps_lines.append(line)

pump_section = "\n".join(pumps_lines)

########################### Valves ################################
## settings = pd.read_parquet(r'data\ditec-wdn\prv_initial_setting-0-static_input.parquet')
##
## settings = settings.iloc[0].T.reset_index()
## settings.columns = ['valve_ID','settings']
## settings = settings[settings['valve_ID'] != 'scenario_id']
## valve_full = pd.merge(settings, pipe, left_on='valve_ID', right_on='pipe_id')
## valve_full['pipe_id'] = valve_full['pipe_id'].str.replace('~@', '')
## print(valve_full.head())
## no enough values (diametre) to create the valve section ####
###############################################################

################### Coordinates ################################

# from generate_coordinates import create_realistic_wdn_layout

# pos, coord_section, G, node_types = create_realistic_wdn_layout(df_merged)
# print(coord_section)

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque
import itertools

# Create graph from merged dataframe
G = nx.Graph()
for _, row in df_merged.iterrows():
    G.add_edge(row['start_node'], row['end_node'])

# Classify nodes based on their prefixes
nodes = set()
for _, row in df_merged.iterrows():
    nodes.add(row['start_node'])
    nodes.add(row['end_node'])

reservoirs = [node for node in nodes if node.startswith('R-') or node in ['R1', 'R2']]
tanks = [node for node in nodes if node.startswith('T-')]
junctions = [node for node in nodes if node.startswith('J-')]
pumps = [node for node in nodes if 'Pump' in node]
valves = [node for node in nodes if '-RV-' in node or node.startswith('I-RV-') or node.startswith('O-RV-')]

print(f"Reservoirs: {len(reservoirs)}, Tanks: {len(tanks)}, Junctions: {len(junctions)}")
print(f"Pumps: {len(pumps)}, Valves: {len(valves)}")

# SUGIYAMA'S HIERARCHICAL LAYOUT ALGORITHM
# Step 1: Create a directed acyclic graph (DAG) for proper hierarchy
root_nodes = reservoirs + tanks
if not root_nodes:
    root_nodes = [max(G.nodes(), key=lambda x: G.degree(x))]

# Build DAG using topological ordering from water sources
DAG = nx.DiGraph()
visited = set()
queue = deque(root_nodes)
visited.update(root_nodes)

# Add all nodes first
for node in G.nodes():
    DAG.add_node(node)

# Build edges following water flow direction
while queue:
    current = queue.popleft()
    neighbors = list(G.neighbors(current))
    
    for neighbor in neighbors:
        if neighbor not in visited:
            DAG.add_edge(current, neighbor)
            visited.add(neighbor)
            queue.append(neighbor)

# Step 2: Layer assignment (longest path layering)
# Assign layers based on longest path from sources
layers = {}
max_layer = 0

# Initialize source nodes at layer 0
for root in root_nodes:
    layers[root] = 0

# Use topological sort to assign layers
topo_order = list(nx.topological_sort(DAG))
for node in topo_order:
    if node not in layers:
        # Find maximum layer of predecessors
        predecessors = list(DAG.predecessors(node))
        if predecessors:
            layers[node] = max(layers[pred] for pred in predecessors) + 1
        else:
            layers[node] = 0
    max_layer = max(max_layer, layers[node])

# Step 3: Add dummy nodes for long edges (edges spanning multiple layers)
# Create extended DAG with dummy nodes
extended_DAG = nx.DiGraph()
dummy_counter = 0

for node in DAG.nodes():
    extended_DAG.add_node(node)

for u, v in DAG.edges():
    layer_diff = layers[v] - layers[u]
    if layer_diff <= 1:
        # Normal edge
        extended_DAG.add_edge(u, v)
    else:
        # Long edge - add dummy nodes
        prev_node = u
        for i in range(1, layer_diff):
            dummy_node = f"dummy_{dummy_counter}"
            dummy_counter += 1
            extended_DAG.add_node(dummy_node)
            extended_DAG.add_edge(prev_node, dummy_node)
            layers[dummy_node] = layers[u] + i
            prev_node = dummy_node
        extended_DAG.add_edge(prev_node, v)

# Step 4: Crossing reduction using barycenter heuristic
# Group nodes by layers
layer_nodes = defaultdict(list)
for node, layer in layers.items():
    layer_nodes[layer].append(node)

# Barycenter crossing reduction
for iteration in range(10):  # Multiple iterations for better results
    # Forward pass (top to bottom)
    for layer in range(1, max_layer + 1):
        nodes_in_layer = layer_nodes[layer]
        if len(nodes_in_layer) <= 1:
            continue
        
        # Calculate barycenter for each node
        barycenters = []
        for node in nodes_in_layer:
            predecessors = list(extended_DAG.predecessors(node))
            if predecessors:
                # Find positions of predecessors in previous layer
                prev_layer_nodes = layer_nodes[layer - 1]
                prev_positions = {n: i for i, n in enumerate(prev_layer_nodes)}
                barycenter = sum(prev_positions.get(pred, 0) for pred in predecessors) / len(predecessors)
            else:
                barycenter = 0
            barycenters.append((barycenter, node))
        
        # Sort by barycenter
        barycenters.sort()
        layer_nodes[layer] = [node for _, node in barycenters]
    
    # Backward pass (bottom to top)
    for layer in range(max_layer - 1, -1, -1):
        nodes_in_layer = layer_nodes[layer]
        if len(nodes_in_layer) <= 1:
            continue
        
        # Calculate barycenter for each node
        barycenters = []
        for node in nodes_in_layer:
            successors = list(extended_DAG.successors(node))
            if successors:
                # Find positions of successors in next layer
                next_layer_nodes = layer_nodes[layer + 1]
                next_positions = {n: i for i, n in enumerate(next_layer_nodes)}
                barycenter = sum(next_positions.get(succ, 0) for succ in successors) / len(successors)
            else:
                barycenter = 0
            barycenters.append((barycenter, node))
        
        # Sort by barycenter
        barycenters.sort()
        layer_nodes[layer] = [node for _, node in barycenters]

# Step 5: Coordinate assignment
pos = {}
width = 2000
height = 1500

for layer in range(max_layer + 1):
    nodes_in_layer = layer_nodes[layer]
    y = height - (layer * height / max(max_layer, 1))
    
    if len(nodes_in_layer) == 1:
        x = width / 2
        pos[nodes_in_layer[0]] = (x, y)
    else:
        # Distribute nodes evenly across width
        for i, node in enumerate(nodes_in_layer):
            x = (i + 1) * width / (len(nodes_in_layer) + 1)
            pos[node] = (x, y)

# Remove dummy nodes from final positions
final_pos = {node: coord for node, coord in pos.items() if not node.startswith('dummy_')}

# Step 6: Fine-tuning with minimal force adjustments
pos_array = {node: np.array(coord) for node, coord in final_pos.items()}

# Light force-directed adjustment to improve aesthetics
for iteration in range(20):
    forces = defaultdict(lambda: np.array([0.0, 0.0]))
    
    # Small repulsive forces to avoid overlaps
    node_list = list(pos_array.keys())
    for i, node1 in enumerate(node_list):
        for node2 in node_list[i+1:]:
            pos1 = pos_array[node1]
            pos2 = pos_array[node2]
            diff = pos2 - pos1
            dist = np.linalg.norm(diff)
            
            if dist > 0 and dist < 100:  # Minimum distance
                direction = diff / dist
                repulsive_force = direction * (100 - dist) * 0.01
                forces[node1] -= repulsive_force
                forces[node2] += repulsive_force
    
    # Update positions with strong damping to preserve hierarchy
    damping = 0.5
    for node in pos_array:
        # Strongly constrain vertical movement to maintain layers
        forces[node][1] *= 0.1
        pos_array[node] += forces[node] * damping

# Convert back to regular dict
pos_final = {node: tuple(coord) for node, coord in pos_array.items()}

# Scale the coordinates
scale = 1000
pos_scaled = {node: (coords[0] * scale / width, coords[1] * scale / height) for node, coords in pos_final.items()}

# Plot the network
plt.figure(figsize=(15, 10))

# Plot edges
nx.draw_networkx_edges(G, pos_scaled, edge_color='gray', alpha=0.6, width=1.5)

# Plot nodes with different colors and sizes based on type
if reservoirs:
    reservoir_pos = {node: pos_scaled[node] for node in reservoirs}
    nx.draw_networkx_nodes(G, reservoir_pos, nodelist=reservoirs, 
                          node_color='blue', node_size=300, node_shape='s', alpha=0.8, label='Reservoirs')

if tanks:
    tank_pos = {node: pos_scaled[node] for node in tanks}
    nx.draw_networkx_nodes(G, tank_pos, nodelist=tanks, 
                          node_color='green', node_size=200, node_shape='^', alpha=0.8, label='Tanks')

if pumps:
    pump_pos = {node: pos_scaled[node] for node in pumps}
    nx.draw_networkx_nodes(G, pump_pos, nodelist=pumps, 
                          node_color='red', node_size=150, node_shape='D', alpha=0.8, label='Pumps')

if valves:
    valve_pos = {node: pos_scaled[node] for node in valves}
    nx.draw_networkx_nodes(G, valve_pos, nodelist=valves, 
                          node_color='purple', node_size=120, node_shape='h', alpha=0.8, label='Valves')

# Plot junctions last (anything not in other categories)
junction_nodes = [node for node in G.nodes() if node not in reservoirs + tanks + pumps + valves]
if junction_nodes:
    junction_pos = {node: pos_scaled[node] for node in junction_nodes}
    nx.draw_networkx_nodes(G, junction_pos, nodelist=junction_nodes, 
                          node_color='orange', node_size=80, node_shape='o', alpha=0.8, label='Junctions')

# Add labels
nx.draw_networkx_labels(G, pos_scaled, font_size=8, font_weight='bold')

plt.title('Water Distribution Network - Sugiyama Hierarchical Layout', fontsize=16, fontweight='bold')
plt.legend(scatterpoints=1, loc='upper right')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Generate coordinates section in the exact format requested
coord_section = '[COORDINATES]\n'
for node, (x, y) in pos_scaled.items():
    coord_section += f'{node} {x:.2f} {y:.2f}\n'

print("Sugiyama algorithm completed successfully!")
print("Preview of coordinates:")
print(coord_section[:300] + "..." if len(coord_section) > 300 else coord_section)


with open(r"data\ditec-wdn\network.inp", "w") as f:
    f.write(junction_section)
    f.write("\n")
    f.write(reservoir_section)
    f.write("\n")
    f.write(tank_section)
    f.write("\n")
    f.write(coord_section)
    f.write("\n")
    f.write(pipe_section)
    f.write("\n")
    f.write(pump_section)
    
    
