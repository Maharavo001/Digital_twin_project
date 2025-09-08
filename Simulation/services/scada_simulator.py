# services/scada_simulator.py
import csv
import time
import numpy as np
from datetime import datetime
from typing import Dict, Generator, Tuple

class ScadaSimulator:
    def __init__(self, demand_csv_path: str, delay_sec: int = 300, 
                 max_steps: int = 48, start_from: int = 0):
        """
        Simulates SCADA data injection from a CSV file or synthetic generator.
        
        Args:
            demand_csv_path: Path to CSV file with demand patterns
            delay_sec: Time between data injections (default: 300s/5min)
            max_steps: Maximum number of rows to stream (real + generated)
            start_from: Index of row to start from (0-based). 
                        -1 = skip all real rows, start from generated.
        """
        self.delay_sec = delay_sec
        self.demand_data = self._load_demand_data(demand_csv_path)
        self.total_real_steps = len(next(iter(self.demand_data.values())))
        self.total_steps = self.total_real_steps
        self.start_time = datetime.now()
        self.max_steps = max_steps
        self.generated_rows = []  # store generated rows

        if start_from == -1:
            # skip all real rows → begin with generation
            self.current_idx = self.total_real_steps
        else:
            if start_from >= self.total_real_steps:
                raise ValueError(
                    f"start_from={start_from} exceeds available real rows={self.total_real_steps}"
                )
            self.current_idx = start_from

    def _load_demand_data(self, csv_path: str) -> Dict[str, list]:
        """Load demand patterns from CSV into a dictionary."""
        demand_data = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for junction, demand in row.items():
                    if junction not in demand_data:
                        demand_data[junction] = []
                    demand_data[junction].append(float(demand))
        return demand_data

    def _get_current_timestamp(self) -> str:
        """Generate formatted timestamp for current simulation time."""
        elapsed = datetime.now() - self.start_time
        return (self.start_time + elapsed).strftime("%Y-%m-%d %H:%M:%S")

    def _generate_next_row(self) -> Dict[str, float]:
        """Generate synthetic demand row if real data is exhausted."""
        next_row = {}
        t = self.current_idx

        for junction, values in self.demand_data.items():
            history = values[-24:] if len(values) >= 24 else values
            base_mean = np.mean(history)
            base_std = np.std(history) if np.std(history) > 0 else 0.1

            daily_cycle = np.cos(2 * np.pi * (t % 24) / 24)
            noise = np.random.normal(0, base_std * 0.1)

            generated = base_mean * (1 + 0.1 * daily_cycle) + noise
            next_row[junction] = max(generated, 0.0)

            self.demand_data[junction].append(next_row[junction])

        self.total_steps += 1
        self.generated_rows.append(next_row)
        return next_row

    def stream_data(self) -> Generator[Tuple[str, Dict[str, float]], None, None]:
        """
        Generator that yields (timestamp, demand_dict) every delay_sec seconds.
        Stops when max_steps is reached.
        """
        while self.current_idx < self.max_steps:
            if self.current_idx < self.total_real_steps:
                current_demands = {
                    junction: values[self.current_idx]
                    for junction, values in self.demand_data.items()
                }
            else:
                current_demands = self._generate_next_row()

            yield (self._get_current_timestamp(), current_demands)

            self.current_idx += 1
            if self.current_idx < self.max_steps:
                time.sleep(self.delay_sec)

            if self.current_idx % 10 == 0:
                print(f"Progress: {self.current_idx}/{self.max_steps} steps")

    def save_dataset(self, output_path: str):
        """Save full dataset (real + generated rows) to CSV with time column."""
        junctions = list(self.demand_data.keys())
        rows = list(zip(*[self.demand_data[j] for j in junctions]))  # transpose

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time"] + junctions)
            for idx, row in enumerate(rows):
                timestamp_val = idx * 100  # 100 = 1h, so 24h=2400
                writer.writerow([timestamp_val] + list(row))

        print(f"Dataset saved to {output_path}")

    def reset(self):
        """Reset the simulator to start from beginning."""
        self.current_idx = 0
        self.start_time = datetime.now()
        self.generated_rows.clear()

