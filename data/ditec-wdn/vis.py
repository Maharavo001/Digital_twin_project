import pandas as pd


junction = pd.read_parquet('data\ditec-wdn\junction_elevation-0-static_input.parquet')
print(junction.columns)