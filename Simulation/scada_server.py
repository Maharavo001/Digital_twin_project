# scada_server.py
import socket
import pickle
import time
from services.scada_simulator import ScadaSimulator

# Initialize SCADA simulator
scada = ScadaSimulator("/mnt/data/home/zayd/Digital_twin_project/machine_learning/dataset/Ctown/test.csv", delay_sec=20, max_steps=48,start_from = 0)


Saved_data_path = "mnt/data/home/zayd/Desktop/Digital_twin_project/machine_learning/dataset/Ctown/augmented_9_3_1.csv"

# Socket setup
HOST = '127.0.0.1'
PORT = 5000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"SCADA Server listening on {HOST}:{PORT}")
    conn, addr = s.accept()
    with conn:
        print(f"Client connected: {addr}")
        for ts, demand in scada.stream_data():
            # Serialize the tuple and send
            data_bytes = pickle.dumps((ts, demand))
            conn.sendall(len(data_bytes).to_bytes(4, 'big') + data_bytes)  # prepend length
            print(f"Sent data for {ts}")
        scada.save_dataset(Saved_data_path)
