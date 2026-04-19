"""
utils/logger.py
Simple CSV logger for detections and alerts.
Logs to output/detections_log.csv for analysis and reporting.
"""

import csv
import os
import time
from datetime import datetime


class Logger:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if not enabled:
            return

        os.makedirs("output", exist_ok=True)
        self.log_file = "output/detections_log.csv"

        # Write CSV header if file is new
        write_header = not os.path.exists(self.log_file)
        self.fh = open(self.log_file, "a", newline="")
        self.writer = csv.writer(self.fh)
        if write_header:
            self.writer.writerow([
                "timestamp", "datetime", "label",
                "distance_m", "confidence", "alert_level"
            ])
        print(f"[Logger] Logging detections to: {self.log_file}")

    def log(self, message: str):
        if self.enabled:
            print(f"[LOG {datetime.now().strftime('%H:%M:%S')}] {message}")

    def log_detections(self, detections: list, alert_level: str):
        if not self.enabled:
            return
        ts = time.time()
        dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for det in detections:
            self.writer.writerow([
                ts, dt,
                det.get("label", ""),
                det.get("distance_m", ""),
                det.get("confidence", ""),
                alert_level
            ])
        self.fh.flush()

    def __del__(self):
        try:
            self.fh.close()
        except Exception:
            pass
