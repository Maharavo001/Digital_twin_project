import os
# Configure environment before other imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging
os.environ['MPLCONFIGDIR'] = '/tmp/'      # Fix matplotlib config

import matplotlib
matplotlib.use('Agg')
import socket
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

# -----------------------------
# Config
# -----------------------------
ABS_TOL = 1e-3     # absolute tolerance
REL_TOL = 0.05     # relative tolerance (5%)
MAX_FAIL_RATIO = 0.3
STEP_DELAY_SEC = 60
HOST_SCADA = '127.0.0.1'
PORT_SCADA = 5000
HOST_RESULT = '127.0.0.1'
PORT_RESULT = 6000

# -----------------------------
# Helper functions
# -----------------------------
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

def is_prediction_acceptable(predicted, real, abs_tol=ABS_TOL, rel_tol=REL_TOL, max_fail_ratio=MAX_FAIL_RATIO):
    total = 0
    failed_nodes = []
    for node, r in real.items():
        if node in predicted:
            total += 1
            p = float(np.array(predicted[node]).squeeze())
            r = float(np.array(r).squeeze())
            abs_error = abs(p - r)
            rel_error = abs_error / (abs(r) + 1e-6)
            if abs_error > abs_tol and rel_error > rel_tol:
                failed_nodes.append(node)
                logger.warning(f"[{node}] abs_error={abs_error:.6f}, rel_error={rel_error:.3%} → FAIL")
            else:
                logger.info(f"[{node}] abs_error={abs_error:.6f}, rel_error={rel_error:.3%} → OK")
    if total == 0:
        return False, failed_nodes, total
    fail_ratio = len(failed_nodes) / total
    logger.info(f"Failed junctions: {len(failed_nodes)}/{total} ({fail_ratio:.1%})")
    return fail_ratio <= max_fail_ratio, failed_nodes, total

# -----------------------------
# Main loop
# -----------------------------
def main():
    predicted_demand = None
    real_state = None
    predicted_state = None
    iteration = -1

    junctions_data_path = "/mnt/data/home/zayd/Digital_twin_project/machine_learning/dataset/Ctown/junctions"
    model_dir = '/mnt/data/home/zayd/Digital_twin_project/machine_learning/model_trained/LightGBM_0.0.5'
    scaler_dir = os.path.join(model_dir, 'scalers')
    dict_path = os.path.join(model_dir, 'feature_and_target.json')
    inp_file = '/mnt/data/home/zayd/Digital_twin_project/inp_networks/ctown_map.inp'

    predictor = ModelManager(
        models_dir=model_dir,
        model_type="lightgbm",
        model_format=".txt",
        feature_scaler_dir=scaler_dir,
        target_scaler_dir=scaler_dir,
        dict_of_cols_and_FE=json.load(open(dict_path, 'r')),
        df_window_path=junctions_data_path,
        seq_length=23
    )

    JUNCTIONS = None  # use all

    # Tracking for summary
    junction_total = {}
    junction_fail = {}
    prediction_accepted = 0
    prediction_rejected = 0
    total_iterations = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as scada_socket, \
         socket.socket(socket.AF_INET, socket.SOCK_STREAM) as result_socket:

        scada_socket.connect((HOST_SCADA, PORT_SCADA))
        logger.info("Connected to SCADA server")

        result_socket.connect((HOST_RESULT, PORT_RESULT))
        logger.info("Connected to Result Receiver")

        for ts, real_demand in receive_scada_data(scada_socket):
            iteration += 1
            total_iterations += 1
            logger.info(f"[{ts}] Step {iteration}")

            filtered_real_demand = real_demand if JUNCTIONS is None else {
                k: v for k, v in real_demand.items() if k in JUNCTIONS
            }

            if iteration == 0:
                logger.info("First iteration - running with real demand...")
                real_results, real_state = run_physical_process(filtered_real_demand, inp_file)

                data_bytes = pickle.dumps(real_results)
                result_socket.sendall(len(data_bytes).to_bytes(4, 'big') + data_bytes)
                logger.info("Sent real_results to receiver")

                predictor.update_window(real_demand)
                predicted_demand = predictor.run_models()
                predicted_results, predicted_state = run_physical_process(predicted_demand, state=real_state)
                continue

            # Compare real vs predicted demand
            ok, failed_nodes, total_nodes = is_prediction_acceptable(predicted_demand, filtered_real_demand)
            for j in filtered_real_demand.keys():
                junction_total[j] = junction_total.get(j, 0) + 1
            for j in failed_nodes:
                junction_fail[j] = junction_fail.get(j, 0) + 1

            if ok:
                prediction_accepted += 1
                real_state = predicted_state
                real_results = predicted_results
            else:
                prediction_rejected += 1
                real_results, real_state = run_physical_process(filtered_real_demand, state=real_state)

            # Update window + next prediction
            predictor.update_window(real_demand)
            predicted_demand = predictor.run_models()
            predicted_results, predicted_state = run_physical_process(predicted_demand, state=real_state)

            # Send results
            data_bytes = pickle.dumps(real_results)
            result_socket.sendall(len(data_bytes).to_bytes(4, 'big') + data_bytes)
            logger.info("Sent real_results to receiver")

        # Summary
        logger.info("=== SIMULATION SUMMARY ===")
        logger.info(f"Total iterations: {total_iterations}")
        logger.info(f"Predictions accepted: {prediction_accepted} ({prediction_accepted/total_iterations:.1%})")
        logger.info(f"Predictions rejected: {prediction_rejected} ({prediction_rejected/total_iterations:.1%})")

        logger.info("Per-junction performance:")
        for j in sorted(junction_total.keys()):
            total = junction_total[j]
            failed = junction_fail.get(j, 0)
            ok_count = total - failed
            logger.info(f"{j}: always correct={ok_count}/{total} ({ok_count/total:.1%}), failed={failed}/{total} ({failed/total:.1%})")

        # shutdown visualization server
        data_bytes = pickle.dumps("END")
        result_socket.sendall(len(data_bytes).to_bytes(4, 'big') + data_bytes)


if __name__ == "__main__":
    main()

