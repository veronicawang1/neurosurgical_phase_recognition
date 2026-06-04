import os
import json
from datetime import datetime


class RunLogger:
    def __init__(self, log_dir, run_name, hparams: dict):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"{run_name}_{timestamp}.json")
        self.data = {"run_name": run_name, "timestamp": timestamp,
                     "hparams": hparams, "epochs": []}
        self._save()
        print(f"Logging to {self.log_path}")

    def log_epoch(self, epoch: int, metrics: dict):
        self.data["epochs"].append({"epoch": epoch, **metrics})
        self._save()

    def _save(self):
        with open(self.log_path, "w") as f:
            json.dump(self.data, f, indent=2)
