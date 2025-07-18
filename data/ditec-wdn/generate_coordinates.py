import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque
import random

def classify_nodes(merged_df):
    """Classify nodes based on their prefixes"""
    nodes = set()
    for _, row in merged_df.iterrows():
        nodes.add(row['start_node'])
        nodes.add(row['end_node'])
    
    node_types = {
        'reservoirs': [node for node in nodes if node.startswith('R-') or node in ['R1', 'R2']],
        'tanks': [node for node in nodes if node.startswith('T-')],
        'junctions': [node for node in nodes if node.startswith('J-')],
        'pumps': [node for node in nodes if 'Pump' in node],
        'valves': [node for node in nodes if '-RV-' in node or node.startswith('I-RV-') or node.startswith('O-RV-')]
    }

    
    return node_types

def create_hierarchical_layout(G, node_types, width=2000, height=1500):
    """Create a hierarchical layout that mimics real WDN topology"""
    pos = {}
    
    # Level 1: Reservoirs at the top (water sources)
    reservoirs = node_types['reservoirs']
    if reservoirs:
        for i, reservoir in enumerate(reservoirs):
            x = (i + 1) * width / (len(reservoirs) + 1)
            y = height * 0.9  # Top 10% of the layout
            pos[reservoir] = (x, y)
    
    # Level 2: Tanks (intermediate storage) - slightly below reservoirs
    tanks = node_types['tanks']
    if tanks:
        for i, tank in enumerate(tanks):
            x = (i + 1) * width / (len(tanks) + 1)
            y = height * 0.75  # 75% height
            pos[tank] = (x, y)
    
    # Level 3: Pumps - positioned strategically between sources and distribution
    pumps = node_types['pumps']
    if pumps:
        for i, pump in enumerate(pumps):
            x = (i + 1) * width / (len(pumps) + 1)
            y = height * 0.6  # 60% height
            pos[pump] = (x, y)
    
        # Level 3.5: Valves - between pumps and junctions (new level)
    valves = node_types.get('valves', [])
    if valves:
        for i, valve in enumerate(valves):
            x = (i + 1) * width / (len(valves) + 1)
            y = height * 0.55  # Slightly below pumps
            pos[valve] = (x, y)

    # Level 4: Junctions - distributed in the lower area (distribution network)
    junctions = node_types['junctions']
    if junctions:
        # Create a more realistic distribution pattern
        # Use a grid-like pattern with some randomness
        n_junctions = len(junctions)
        
        # Calculate grid dimensions
        cols = int(np.sqrt(n_junctions * 1.5))  # Slightly wider than square
        rows = int(np.ceil(n_junctions / cols))
        
        junction_positions = []
        for i in range(rows):
            for j in range(cols):
                if len(junction_positions) >= n_junctions:
                    break
                
                # Base grid position
                x = (j + 1) * width / (cols + 1)
                y = height * 0.5 - (i * height * 0.4 / max(rows, 1))  # Distribute in lower 50%
                
                # Add some randomness to avoid perfect grid
                x += random.uniform(-width/20, width/20)
                y += random.uniform(-height/30, height/30)
                
                junction_positions.append((x, y))
        
        for i, junction in enumerate(junctions):
            if i < len(junction_positions):
                pos[junction] = junction_positions[i]
    
    return pos

def improve_layout_with_connectivity(G, pos, node_types, iterations=50):
    """Improve layout by considering network connectivity and water flow logic"""
    
    # Create a copy of positions for modification
    improved_pos = pos.copy()
    
    for iteration in range(iterations):
        forces = defaultdict(lambda: np.array([0.0, 0.0]))
        
        # Apply forces based on network topology
        for edge in G.edges():
            node1, node2 = edge
            if node1 in improved_pos and node2 in improved_pos:
                # Calculate current distance
                pos1 = np.array(improved_pos[node1])
                pos2 = np.array(improved_pos[node2])
                diff = pos2 - pos1
                dist = np.linalg.norm(diff)
                
                if dist > 0:
                    # Normalize direction
                    direction = diff / dist
                    
                    # Ideal distance based on node types
                    ideal_dist = calculate_ideal_distance(node1, node2, node_types)
                    
                    # Apply spring force
                    force_magnitude = (dist - ideal_dist) * 0.01
                    force = direction * force_magnitude
                    
                    forces[node1] += force
                    forces[node2] -= force
        
        # Apply repulsive forces between nodes of same type
        for node_type, nodes in node_types.items():
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    if node1 in improved_pos and node2 in improved_pos:
                        pos1 = np.array(improved_pos[node1])
                        pos2 = np.array(improved_pos[node2])
                        diff = pos2 - pos1
                        dist = np.linalg.norm(diff)
                        
                        if dist > 0 and dist < 300:  # Only apply if too close
                            direction = diff / dist
                            repulsive_force = direction * (300 - dist) * 0.005
                            forces[node1] -= repulsive_force
                            forces[node2] += repulsive_force
        
        # Update positions
        for node in improved_pos:
            improved_pos[node] = (
                improved_pos[node][0] + forces[node][0],
                improved_pos[node][1] + forces[node][1]
            )
    
    return improved_pos

def calculate_ideal_distance(node1, node2, node_types):
    """Calculate ideal distance between two nodes based on their types"""
    
    # Get node types
    type1 = get_node_type(node1, node_types)
    type2 = get_node_type(node2, node_types)
    
    # Define ideal distances based on network hierarchy
    distance_matrix = {
        ('reservoirs', 'tanks'): 200,
        ('reservoirs', 'pumps'): 250,
        ('reservoirs', 'junctions'): 400,
        ('tanks', 'pumps'): 150,
        ('tanks', 'junctions'): 300,
        ('pumps', 'junctions'): 200,
        ('junctions', 'junctions'): 150,
    }
    
    # Get ideal distance (order doesn't matter)
    key = tuple(sorted([type1, type2]))
    return distance_matrix.get(key, 200)

def get_node_type(node, node_types):
    """Get the type of a node"""
    for node_type, nodes in node_types.items():
        if node in nodes:
            return node_type
    return 'unknown'

def plot_realistic_wdn(G, pos, node_types, title="Realistic Water Distribution Network"):
    """Plot the network with proper styling for different node types"""
    
    plt.figure(figsize=(15, 10))
    
    # Define colors and sizes for different node types
    node_styles = {
        'reservoirs': {'color': 'blue', 'size': 300, 'shape': 's', 'label': 'Reservoirs'},
        'tanks': {'color': 'green', 'size': 200, 'shape': '^', 'label': 'Tanks'},
        'pumps': {'color': 'red', 'size': 150, 'shape': 'D', 'label': 'Pumps'},
        'junctions': {'color': 'orange', 'size': 80, 'shape': 'o', 'label': 'Junctions'}
    }
    
    # Plot edges first
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.6, width=1.5)
    
    # Plot nodes by type
    for node_type, nodes in node_types.items():
        if nodes and node_type in node_styles:
            style = node_styles[node_type]
            node_pos = {node: pos[node] for node in nodes if node in pos}
            
            nx.draw_networkx_nodes(G, node_pos, 
                                 nodelist=nodes,
                                 node_color=style['color'],
                                 node_size=style['size'],
                                 node_shape=style['shape'],
                                 alpha=0.8,
                                 label=style['label'])
    
    # Add labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.legend(scatterpoints=1, loc='upper right')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def generate_coordinates_section(pos):
    """Generate the COORDINATES section for EPANET input file"""
    coord_section = '[COORDINATES]\n'
    for node, (x, y) in pos.items():
        coord_section += f'{node} {x:.2f} {y:.2f}\n'
    
    return coord_section

# Main execution function
def create_realistic_wdn_layout(merged_df):
    """Main function to create realistic WDN layout"""
    
    # Create graph from merged dataframe
    G = nx.Graph()
    for _, row in merged_df.iterrows():
        G.add_edge(row['start_node'], row['end_node'])
    
    # Classify nodes
    node_types = classify_nodes(merged_df)
    
    print("Node Classification:")
    for node_type, nodes in node_types.items():
        print(f"{node_type.capitalize()}: {len(nodes)} nodes")
    
    # Create hierarchical layout
    pos = create_hierarchical_layout(G, node_types)
    
    # Improve layout with connectivity considerations
    pos_improved = improve_layout_with_connectivity(G, pos, node_types)
    
    # Plot the network
    plot_realistic_wdn(G, pos_improved, node_types)
    
    # Generate coordinates section
    coord_section = generate_coordinates_section(pos_improved)
    
    return pos_improved, coord_section, G, node_types
