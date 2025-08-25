import socket
import pickle
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

HOST = '127.0.0.1'
PORT = 6000

def main():
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

                logger.info(f"Received real_results: {real_results}")

if __name__ == "__main__":
    main()

