import socket
import pickle
import logging
from influxdb import InfluxDBClient
import yaml

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Configuration from YAML
HOST = config['mininet']['hosts']['h5']['ip'].split('/')[0]  # '10.0.0.5'
PORT = config['network']['result_port']                       # 6000

INFLUX_HOST = config['influxdb']['host']
INFLUX_PORT = config['influxdb']['port']
INFLUX_USER = config['influxdb']['user']
INFLUX_PASSWORD = config['influxdb']['password']
INFLUX_DB = config['influxdb']['database']

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------
# InfluxDB helper
# -----------------------------
def init_influxdb(host, port, username, password, database):
    client = InfluxDBClient(host=host, port=port, username=username, password=password)
    databases = [db['name'] for db in client.get_list_database()]
    if database not in databases:
        client.create_database(database)
        logger.info(f"Created InfluxDB database: {database}")
    client.switch_database(database)
    return client

# -----------------------------
# Main receiver
# -----------------------------
def main():
    influx_client = init_influxdb(INFLUX_HOST, INFLUX_PORT, INFLUX_USER, INFLUX_PASSWORD, INFLUX_DB)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        logger.info(f"Result receiver listening on {HOST}:{PORT}")
        
        conn, addr = s.accept()
        with conn:
            logger.info(f"Client connected: {addr}")
            while True:
                # Read 4-byte length prefix
                len_bytes = conn.recv(4)
                if not len_bytes:
                    continue  # keep waiting instead of breaking

                data_len = int.from_bytes(len_bytes, 'big')

                # Receive actual data
                data_bytes = b""
                while len(data_bytes) < data_len:
                    packet = conn.recv(data_len - len(data_bytes))
                    if not packet:
                        break
                    data_bytes += packet

                # Deserialize
                real_results = pickle.loads(data_bytes)

                # Check for end signal
                if real_results == "END":
                    logger.info("Received END signal. Shutting down receiver.")
                    break

                # Prepare points for InfluxDB
                points = []
                iteration = real_results.get("iteration", 0) if isinstance(real_results, dict) else 0

                # Tanks
                for tank, value in real_results.get("tanks", {}).items():
                    points.append({
                        "measurement": "tanks",
                        "tags": {"id": tank},
                        "fields": {"value": float(value), "iteration": iteration}
                    })
                # Junctions
                for junction, value in real_results.get("junctions", {}).items():
                    points.append({
                        "measurement": "junctions",
                        "tags": {"id": junction},
                        "fields": {"value": float(value), "iteration": iteration}
                    })
                # Pumps
                for pump, stats in real_results.get("pumps", {}).items():
                    points.append({
                        "measurement": "pumps",
                        "tags": {"id": pump},
                        "fields": {"flow": float(stats.get("flow", 0.0)),
                                   "status": float(stats.get("status", 0)),
                                   "iteration": iteration}
                    })
                # Valves
                for valve, stats in real_results.get("valves", {}).items():
                    points.append({
                        "measurement": "valves",
                        "tags": {"id": valve},
                        "fields": {"flow": float(stats.get("flow", 0.0)),
                                   "status": float(stats.get("status", 0)),
                                   "iteration": iteration}
                    })

                # Write to InfluxDB
                if points:
                    influx_client.write_points(points)
                    logger.info(f"Data sent to InfluxDB for iteration {iteration}")

if __name__ == "__main__":
    main()
