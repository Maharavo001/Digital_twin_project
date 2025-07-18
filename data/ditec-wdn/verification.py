import pandas as pd

# Load the pipe connectivity file
df = pd.read_csv('data\ditec-wdn\pipe_connectivity.csv')

# Collect unique nodes
nodes = set(df['start_node']).union(set(df['end_node']))

# Classify node types
junctions = [n for n in nodes if n.startswith('J-')]
valves = [n for n in nodes if '-RV-' in n or n.startswith('I-RV-') or n.startswith('O-RV-')]
reservoirs = [n for n in nodes if n.startswith('R-') or n in ['R1', 'R2']]
tanks = [n for n in nodes if n.startswith('T-')]
raw_pumps = [n for n in nodes if '-Pump-' in n]
pumps = set(p.split('-Pump-')[-1] for p in raw_pumps)


# Print counts
print("Junctions:", len(junctions), "(Expected: 920)")
print("Pipes:", len(df), "(Expected: 1061)")
print("Reservoirs:", len(reservoirs), "(Expected: 2)")
print("Tanks:", len(tanks), "(Expected: 13)")
print("Pumps:", len(pumps), "(Expected: 13)")
print(f"Valves: {len(valves)}")


classified = set(junctions + reservoirs + tanks + raw_pumps + valves)
unclassified = sorted(set(nodes) - classified)

print(f"\nUnclassified nodes ({len(unclassified)}):")
for node in unclassified:
    print(node)


