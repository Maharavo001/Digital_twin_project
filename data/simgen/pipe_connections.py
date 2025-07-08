import json
import pandas as pd

# Load JSON
with open(r"C:\Users\warag\OneDrive\Bureau\Digital_twin_project\data\simgen\attrs.json", "r") as f:
    data = json.load(f)

# Extract pipes
pipes = data['adj_list']

# Create list of pipe records
pipe_records = []
for pipe in pipes:
    start, end, pipe_id = pipe
    pipe_records.append({
        'start_node': start,
        'end_node': end,
        'pipe_id': pipe_id
    })

# Convert to DataFrame
df = pd.DataFrame(pipe_records)

df.to_csv("pipe_connections.csv", index=False)

