import subprocess
import json

def run_script(input_arg, demand_json):
    """Run the physical_process.py script with given args and return its output."""
    result = subprocess.run(
        ["python", "physical_process.py", input_arg, json.dumps(demand_json)],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Script failed: {result.stderr or result.stdout}")
    return json.loads(result.stdout)

def test_pipeline():
    # 1. Start with inp file
    first = run_script("minitown_map.inp", {"J1": 10})
    print("First run results:", first["results"])

    # 2. Use returned state for next run
    state_json = json.dumps(first["state"])
    second = run_script(state_json, {"J2": 15})
    print("Second run results:", second["results"])

    # 3. Show new state
    print("Final state:", second["state"])

if __name__ == "__main__":
    test_pipeline()
