import os
import time
import subprocess
import signal
import shutil

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

processes = {}

# Virtual environment Python
VENV_PYTHON = "/home/zayd/Desktop/Digital_twin_project/digitaltwin/bin/python"

PORTS_TO_FREE = [5000,6000] 
def free_ports():
    for port in PORTS_TO_FREE:
        try:
            # This kills any process using the port
            subprocess.run(["sudo", "fuser", "-k", f"{port}/tcp"], check=False)
            print(f"Freed port {port}")
        except Exception as e:
            print(f"Could not free port {port}: {e}")


# Check if the Python path exists
if not shutil.which(VENV_PYTHON):
    raise FileNotFoundError(f"Python executable not found: {VENV_PYTHON}")

def run_in_terminal(script, log_name):
    log_path = os.path.abspath(os.path.join(LOG_DIR, log_name))

    # Overwrite logs instead of appending
    cmd = (
        f'while true; do '
        f'echo "Starting {script}..."; '
        f'"{VENV_PYTHON}" "{script}" 2>&1 | tee "{log_path}"; '  # removed -a
        f'if [ $? -ne 0 ]; then echo "{script} failed, exiting loop."; break; fi; '
        f'echo "{script} crashed. Restarting in 3s..."; sleep 3; '
        f'done'
    )
    return subprocess.Popen([
        "gnome-terminal", "--", "bash", "-c", f"{cmd}; exec bash"
    ])

if __name__ == "__main__":
    try:
        processes["scada"] = run_in_terminal("scada_server.py", "scada.log")
        print("Opened SCADA server terminal.")

        processes["visualization"] = run_in_terminal("visualization_server.py", "visualization.log")
        print("Opened Visualization server terminal.")

        time.sleep(7)  # let servers initialize

        processes["controller"] = run_in_terminal("real-time_controller.py", "controller.log")
        print("Opened Real-time controller terminal.")

        signal.pause()  # Wait forever

    except KeyboardInterrupt:
        print("\nStopping all processes...")
        for name, proc in processes.items():
            if proc.poll() is None:
                proc.terminate()
        free_ports()  # free occupied ports
        print("Shutdown complete.")

