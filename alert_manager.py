"""
core/alert_manager.py
Evaluates detected objects and triggers audio + visual alerts
based on distance thresholds.

Alert levels:
    SAFE    — all objects beyond warning distance
    WARNING — at least one object within warning zone
    DANGER  — at least one object within danger zone
"""

import threading
import time
import platform


class AlertManager:
    def __init__(self, audio_enabled: bool = True):
        self.audio_enabled  = audio_enabled
        self._alert_thread  = None
        self._last_alert_t  = 0
        self._alert_cooldown = 1.5   # seconds between audio alerts
        self.current_level  = "SAFE"
        print("[AlertManager] Ready. Audio:", "ON" if audio_enabled else "OFF")

    def evaluate(self, detections: list, danger_dist: float, warning_dist: float) -> str:
        """
        Check all detections and return the highest alert level.
        Also triggers audio beep on DANGER/WARNING.
        """
        level = "SAFE"

        for det in detections:
            d = det.get("distance_m", 99)
            if d <= danger_dist:
                level = "DANGER"
                break
            elif d <= warning_dist:
                if level != "DANGER":
                    level = "WARNING"

        self.current_level = level

        # Trigger audio if alert level changed or cooldown passed
        now = time.time()
        if level != "SAFE" and (now - self._last_alert_t) > self._alert_cooldown:
            self._last_alert_t = now
            if self.audio_enabled:
                t = threading.Thread(target=self._beep, args=(level,), daemon=True)
                t.start()

        return level

    def _beep(self, level: str):
        """Cross-platform audio alert."""
        try:
            if platform.system() == "Windows":
                import winsound
                if level == "DANGER":
                    winsound.Beep(1000, 300)
                    time.sleep(0.1)
                    winsound.Beep(1000, 300)
                else:
                    winsound.Beep(700, 200)
            else:
                # Linux / Mac fallback — use system bell
                print("\a", end="", flush=True)
        except Exception as e:
            pass   # Silent fail — audio is non-critical
