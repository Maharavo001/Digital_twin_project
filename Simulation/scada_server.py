import socket
import pickle
import time
import yaml
import logging
from services.scada_simulator import ScadaSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

logger.info("=" * 50)
logger.info("SCADA SIMULATION SERVER STARTING")
logger.info("=" * 50)

# Load configuration
logger.info("Loading configuration from config.yaml...")
try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    exit(1)

# Configuration from YAML
test_csv = config['paths']['test_csv']
delay_sec = config['simulation']['step_delay_sec']
max_steps = config['simulation']['max_steps']
start_from = config['simulation']['start_from']
saved_data_path = config['paths']['augmented_csv']
# For SCADA connecting to the simulation server
HOST = config['mininet']['hosts']['h1']['ip'].split('/')[0]  # '10.0.0.2'
PORT = config['network']['scada_port']                        # 5000

logger.info(f"Host: {HOST}")
logger.info(f"Port: {PORT}")

# Initialize SCADA simulator
logger.info("Initializing SCADA simulator...")
try:
    scada = ScadaSimulator(test_csv, delay_sec=delay_sec, max_steps=max_steps, start_from=start_from)
    logger.info("SCADA simulator initialized successfully")
except Exception as e:
    logger.error(f"Error initializing SCADA simulator: {e}")
    exit(1)

logger.info(f"Starting socket server on {HOST}:{PORT}...")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    try:
        s.bind((HOST, PORT))
        s.listen()
        logger.info(f"SCADA Server listening on {HOST}:{PORT}")
        logger.info("Waiting for client connection...")
    except Exception as e:
        logger.error(f"Error starting socket server: {e}")
        exit(1)
        
    conn, addr = s.accept()
    with conn:
        logger.info(f"Client connected: {addr}")
        logger.info("Starting data streaming...")
        
        data_count = 0
        for ts, demand in scada.stream_data():
            # Serialize the tuple and send
            data_bytes = pickle.dumps((ts, demand))
            conn.sendall(len(data_bytes).to_bytes(4, 'big') + data_bytes)  # prepend length
            logger.info(f"Sent data for timestamp: {ts}")
            data_count += 1
            
        logger.info(f"Data streaming completed. Total data points sent: {data_count}")
        
        logger.info("Saving dataset...")
        try:
            scada.save_dataset(saved_data_path)
            logger.info(f"Dataset saved successfully to: {saved_data_path}")
        except Exception as e:
            logger.error(f"Error saving dataset: {e}")

logger.info("=" * 50)
logger.info("SCADA SIMULATION SERVER SHUTDOWN COMPLETE")
logger.info("=" * 50)
