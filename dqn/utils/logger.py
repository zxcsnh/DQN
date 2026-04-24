from __future__ import annotations

import csv
from pathlib import Path


class CSVLogger:
    def __init__(self, file_path: Path) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = [
            "episode",
            "total_reward",
            "steps",
            "epsilon",
            "loss",
            "success",
            "custom_metric",
            "eval_avg_reward",
            "eval_avg_steps",
            "eval_success_rate",
            "eval_avg_custom_metric",
            "is_best_model",
        ]
        with self.file_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, row: dict) -> None:
        missing_fields = [field for field in self.fieldnames if field not in row]
        if missing_fields:
            raise ValueError(f"日志字段缺失: {missing_fields}")
        with self.file_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.fieldnames)
            writer.writerow(row)
