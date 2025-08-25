import os
# Configure environment before other imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['MPLCONFIGDIR'] = '/tmp/'      # Fix matplotlib config

import matplotlib
matplotlib.use('Agg')
import socket
import time
import subprocess
import json
import numpy as np
import pickle
from services.prediction_service import ModelManager
import logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


LOSS_THRESHOLD = 5
STEP_DELAY_SEC = 60 #second
HOST_SCADA = '127.0.0.1'
PORT_SCADA = 5000
HOST_RESULT = '127.0.0.1'
PORT_RESULT = 6000

def calculate_loss(predicted: dict, real: dict) -> float:
    """
    Calculate average relative error across all junctions.
    Only considers junctions present in predicted.
    """
    losses = []
    for node, r in real.items():
        if node in predicted:
            p = predicted[node]
            # Ensure scalar
            p = float(p) if isinstance(p, (list, np.ndarray)) else p
            r = float(r) if isinstance(r, (list, np.ndarray)) else r
            if r != 0:
                losses.append(abs(p - r) / r)
    if not losses:
        return float('inf')
    return sum(losses) / len(losses)
    
def receive_scada_data(s):
    """Generator to receive (timestamp, demand_dict) from server"""
    while True:
        len_bytes = s.recv(4)
        if not len_bytes:
            break
        data_len = int.from_bytes(len_bytes, 'big')
        data_bytes = b""
        while len(data_bytes) < data_len:
            packet = s.recv(data_len - len(data_bytes))
            if not packet:
                break
            data_bytes += packet
        ts, demand = pickle.loads(data_bytes)
        yield ts, demand

def run_physical_process(demand_dict, state):
    """Run one step with demand input and state persistence"""
    # Convert dict to JSON string if needed
    state_arg = state if isinstance(state, str) else json.dumps(state)

    cmd = [
        'python',
        'physical_process.py',
        state_arg,
        json.dumps(demand_dict)
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"Physical process failed:\nSTDOUT: {stdout}\nSTDERR: {stderr}")

    output = json.loads(stdout)

    if "error" in output:
        raise RuntimeError(output["error"])

    return output["results"], output["state"]

def main():
    predicted_demand = None
    real_state = None
    predicted_state = None
    iteration = -1
    junctions_data_path = "/home/zayd/Desktop/Digital_twin_project/machine_learning/dataset/junctions"
    model_dir ='/home/zayd/Desktop/Digital_twin_project/machine_learning/model_trained/LightGBM_0.0.1'
    scaler_dir = os.path.join(model_dir,'scalers')
    dict_path = os.path.join(model_dir,'feature_and_target.json')
    inp_file = '/home/zayd/Desktop/Digital_twin_project/enhanced_ctown_topology/ctown_map.inp'
    
    #scada = ScadaSimulator(Dataset_path, delay_sec=STEP_DELAY_SEC)
    
    predictor = ModelManager(models_dir= model_dir, 
                 model_type = "lightgbm",
                 model_format = '.txt',
                 feature_scaler_dir= scaler_dir, 
                 target_scaler_dir= scaler_dir,
                 dict_of_cols_and_FE=json.load(open(dict_path, 'r')),
                 df_window_path= junctions_data_path,
                 seq_length= 23)   
    
    # Define which junctions to include (can be all, or a subset)
    JUNCTIONS = None  
   
       # Connect to SCADA server
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as scada_socket, \
         socket.socket(socket.AF_INET, socket.SOCK_STREAM) as result_socket:
         
        scada_socket.connect((HOST_SCADA, PORT_SCADA))
        logger.info("Connected to SCADA server")
        
        result_socket.connect((HOST_RESULT, PORT_RESULT))
        logger.info("Connected to Result Receiver")
   
        for ts, real_demand in receive_scada_data(scada_socket):
            logger.info("Predicting next step...")
            iteration += 1
            logger.info(f"[{ts}] Step {iteration}...")

            # Select junctions dynamically
            if JUNCTIONS is None:
                filtered_real_demand = real_demand  # use all
            else:
                filtered_real_demand = {k: v for k, v in real_demand.items() if k in JUNCTIONS}

            if iteration == 0:
                logger.info("First iteration - running with real demand...")
                real_results, real_state = run_physical_process(filtered_real_demand, inp_file)
                
                data_bytes = pickle.dumps(real_results)
                result_socket.sendall(len(data_bytes).to_bytes(4, 'big') + data_bytes)
                logger.info("Sent real_results to receiver")
                
                predictor.update_window(real_demand)
                logger.info(f"Predicted demand for next step...")
                predicted_demand = predictor.run_models()
                predicted_results, predicted_state = run_physical_process(predicted_demand, state=real_state)
                logger.info("done")
                continue

            logger.info("Comparing previous prediction with current real demand ...")
            loss = calculate_loss(predicted_demand, real_demand)
            logger.info(f"Prediction loss: {float(loss):.4f} {'(ACCEPTED)' if float(loss) <= LOSS_THRESHOLD else '(REJECTED)'}")

            if loss <= LOSS_THRESHOLD:
                logger.info("Running with previous accepted prediction")
                real_state = predicted_state
                real_results = predicted_results
            else:
                logger.warning("Prediction rejected - running with real demand ...")
                real_results, real_state = run_physical_process(filtered_real_demand, state=real_state)

            predictor.update_window(real_demand)
            predicted_demand = predictor.run_models()
            logger.info(f"Predicted demand for next step done ...")
            predicted_results, predicted_state = run_physical_process(predicted_demand, state=real_state)
            
            data_bytes = pickle.dumps(real_results)
            result_socket.sendall(len(data_bytes).to_bytes(4, 'big') + data_bytes)
            logger.info("Sent real_results to receiver")
        #shutdown visualization server
        data_bytes = pickle.dumps("END")
        result_socket.sendall(len(data_bytes).to_bytes(4, 'big') + data_bytes)
if __name__ == "__main__":
    main()




