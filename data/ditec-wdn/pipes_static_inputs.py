import pandas as pd
from functools import reduce

main = pd.read_csv('data\ditec-wdn\pipe_connectivity.csv')

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



dfs = [main, Length, Diameter, Roughness, MinorLoss, Status]  # list of all your DataFrames
df_merged = main.merge(Length, on='pipe_id', how='left')
df_merged =  df_merged.merge(Diameter, on='pipe_id', how='left')
df_merged =  df_merged.merge(Roughness, on='pipe_id', how='left')
df_merged =  df_merged.merge(MinorLoss, on='pipe_id', how='left')
df_merged =  df_merged.merge(Status, on='pipe_id', how='left')


df_merged.to_csv('data\ditec-wdn\pipes_static_inputs.csv')