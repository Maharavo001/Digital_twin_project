import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

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
valves =  [node for node in nodes if '-RV-' in node or node.startswith('I-RV-') or node.startswith('O-RV-')]

print(f"Reservoirs: {len(reservoirs)}, Tanks: {len(tanks)}, Junctions: {len(junctions)}, Pumps: {len(pumps)} , valves: {len(valves)}")

# Find root nodes (reservoirs and tanks as water sources)
root_nodes = reservoirs + tanks + valves
if not root_nodes:
    # If no reservoirs/tanks, use nodes with highest degree
    root_nodes = [max(G.nodes(), key=lambda x: G.degree(x))]

# Create a directed graph for tree layout with proper water flow direction
DG = nx.DiGraph()

# Use BFS to create tree structure from root nodes
visited = set()
queue = root_nodes.copy()
visited.update(root_nodes)

# Add root nodes to directed graph
for root in root_nodes:
    DG.add_node(root)

# BFS to build tree structure
while queue:
    current = queue.pop(0)
    neighbors = list(G.neighbors(current))
    
    for neighbor in neighbors:
        if neighbor not in visited:
            # Add edge in flow direction (from source towards distribution)
            DG.add_edge(current, neighbor)
            visited.add(neighbor)
            queue.append(neighbor)

# Handle disconnected components
for node in G.nodes():
    if node not in visited:
        DG.add_node(node)

# Use hierarchical layout (similar to Reingold-Tilford)
pos = nx.nx_agraph.graphviz_layout(DG, prog='dot', root=root_nodes[0] if root_nodes else None)

# If graphviz not available, use alternative hierarchical layout
if not pos:
    # Manual hierarchical layout
    pos = {}
    
    # Level-based positioning
    levels = {}
    max_level = 0
    
    # Assign levels using BFS from root nodes
    for root in root_nodes:
        queue = [(root, 0)]
        while queue:
            node, level = queue.pop(0)
            if node not in levels:
                levels[node] = level
                max_level = max(max_level, level)
                for successor in DG.successors(node):
                    queue.append((successor, level + 1))
    
    # Assign remaining nodes to levels
    for node in DG.nodes():
        if node not in levels:
            levels[node] = max_level + 1
    
    # Position nodes level by level
    level_nodes = defaultdict(list)
    for node, level in levels.items():
        level_nodes[level].append(node)
    
    width = 2000
    height = 1500
    
    for level, nodes_in_level in level_nodes.items():
        y = height - (level * height / (max_level + 1))
        for i, node in enumerate(nodes_in_level):
            x = (i + 1) * width / (len(nodes_in_level) + 1)
            pos[node] = (x, y)

# Convert to numpy arrays for easier manipulation
pos_array = {node: np.array(coord) for node, coord in pos.items()}

# Apply some spreading to avoid overlaps
for iteration in range(30):
    forces = defaultdict(lambda: np.array([0.0, 0.0]))
    
    # Repulsive forces between nodes
    node_list = list(pos_array.keys())
    for i, node1 in enumerate(node_list):
        for node2 in node_list[i+1:]:
            pos1 = pos_array[node1]
            pos2 = pos_array[node2]
            diff = pos2 - pos1
            dist = np.linalg.norm(diff)
            
            if dist > 0 and dist < 150:  # Minimum distance
                direction = diff / dist
                repulsive_force = direction * (150 - dist) * 0.1
                forces[node1] -= repulsive_force
                forces[node2] += repulsive_force
    
    # Update positions
    for node in pos_array:
        pos_array[node] += forces[node]

# Convert back to regular dict
pos_final = {node: tuple(coord) for node, coord in pos_array.items()}

# Scale the coordinates
scale = 1000
pos_scaled = {node: (coords[0]*scale/2000, coords[1]*scale/1500) for node, coords in pos_final.items()}

# Plot the network
plt.figure(figsize=(15, 10))

# Plot edges
nx.draw_networkx_edges(G, pos_scaled, edge_color='gray', alpha=0.6, width=1.5)

# Plot nodes with different colors and sizes
node_colors = []
node_sizes = []
for node in G.nodes():
    if node in reservoirs:
        node_colors.append('blue')
        node_sizes.append(300)
    elif node in tanks:
        node_colors.append('green')
        node_sizes.append(200)
    elif node in pumps:
        node_colors.append('red')
        node_sizes.append(150)
    else:  # junctions
        node_colors.append('orange')
        node_sizes.append(80)

nx.draw_networkx_nodes(G, pos_scaled, node_color=node_colors, node_size=node_sizes, alpha=0.8)
nx.draw_networkx_labels(G, pos_scaled, font_size=8, font_weight='bold')

plt.title('Water Distribution Network - Hierarchical Layout', fontsize=16, fontweight='bold')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Generate coordinates section in the exact format requested
coord_section = '[COORDINATES]\n'
for node, (x, y) in pos_scaled.items():
    coord_section += f'{node} {x:.2f} {y:.2f}\n'

print("Coordinates section generated successfully!")
print("Preview:")
print(coord_section[:200] + "..." if len(coord_section) > 200 else coord_section)