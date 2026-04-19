"""
BlindSpot AI - Driver Assistance System
Final Year Project
Author: SAURABH KALUGADE
Description: Real-time blind spot and obstacle detection using MiDaS depth
             estimation + YOLOv8 object detection with audio alerts and UI dashboard.
"""

import cv2
import torch
import numpy as np
import time
import threading
from core.depth_estimator import DepthEstimator
from core.object_detector import ObjectDetector
from core.distance_calculator import DistanceCalculator
from core.alert_manager import AlertManager
from ui.dashboard import Dashboard
from utils.logger import Logger

# ─── Configuration ────────────────────────────────────────────────────────────
CONFIG = {
    "camera_index": 0,
    "frame_width": 640,
    "frame_height": 480,
    "danger_distance_m": 2.0,      # metres — triggers red alert
    "warning_distance_m": 4.0,     # metres — triggers yellow warning
    "focal_length_px": 600,        # calibrate for your camera
    "known_object_width_m": 0.45,  # average person shoulder width
    "enable_audio": True,
    "enable_logging": True,
    "yolo_model": "yolov8n.pt",    # nano model — fast on CPU
    "depth_model": "DPT_Hybrid",
    "target_classes": [            # YOLO classes to detect
        "person", "car", "truck", "bus",
        "motorcycle", "bicycle", "traffic light", "stop sign"
    ],
}
# ──────────────────────────────────────────────────────────────────────────────


def main():
    print("=" * 55)
    print("  BlindSpot AI — Driver Assistance System")
    print("  Starting up... please wait..")
    print("=" * 55)

    logger = Logger(enabled=CONFIG["enable_logging"])
    logger.log("System starting")

    # Initialise all modules
    depth_estimator = DepthEstimator(CONFIG["depth_model"])
    object_detector = ObjectDetector(CONFIG["yolo_model"], CONFIG["target_classes"])
    distance_calc   = DistanceCalculator(CONFIG["focal_length_px"], CONFIG["known_object_width_m"])
    alert_manager   = AlertManager(CONFIG["enable_audio"])
    dashboard       = Dashboard(CONFIG["frame_width"], CONFIG["frame_height"])

    cap = cv2.VideoCapture(CONFIG["camera_index"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])

    if not cap.isOpened():
        print("[ERROR] Could not open camera. Check CONFIG['camera_index'].")
        return

    print("\n[OK] Camera opened")
    print("[OK] All modules loaded")
    print("\nControls:  ESC or Q = Quit  |  S = Screenshot  |  D = Toggle depth map")
    print("-" * 55)

    show_depth = False
    frame_count = 0
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        t_start = time.time()
        frame_count += 1

        # ── 1. Object Detection (YOLO) ──────────────────────────────────────
        detections = object_detector.detect(frame)

        # ── 2. Depth Estimation (MiDaS) ─────────────────────────────────────
        depth_map, depth_colormap = depth_estimator.estimate(frame)

        # ── 3. Distance Calculation ──────────────────────────────────────────
        objects_with_distance = distance_calc.calculate(detections, depth_map, frame.shape)

        # ── 4. Alert Logic ───────────────────────────────────────────────────
        alert_level = alert_manager.evaluate(
            objects_with_distance,
            CONFIG["danger_distance_m"],
            CONFIG["warning_distance_m"]
        )

        # ── 5. Draw UI Dashboard ─────────────────────────────────────────────
        t_end = time.time()
        fps_display = 1 / max(t_end - t_start, 1e-6)

        output_frame = dashboard.render(
            frame=frame,
            detections=objects_with_distance,
            depth_colormap=depth_colormap if show_depth else None,
            alert_level=alert_level,
            fps=fps_display,
            frame_count=frame_count
        )

        cv2.imshow("BlindSpot AI — Driver Assistance", output_frame)

        # ── 6. Logger ────────────────────────────────────────────────────────
        if objects_with_distance:
            logger.log_detections(objects_with_distance, alert_level)

        # ── Key Handling ─────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):   # ESC or Q
            print("\n[INFO] Exiting...")
            break
        elif key == ord('s'):
            fname = f"output/screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, output_frame)
            print(f"[INFO] Screenshot saved: {fname}")
        elif key == ord('d'):
            show_depth = not show_depth
            print(f"[INFO] Depth map: {'ON' if show_depth else 'OFF'}")

    cap.release()
    cv2.destroyAllWindows()
    logger.log("System stopped")
    print("Goodbye..! Have a Nice Day..!!")


if __name__ == "__main__":
    main()
