"""
core/object_detector.py
YOLOv8 object detection wrapper.
Detects road-relevant objects: cars, people, trucks, motorcycles, etc.
"""

import numpy as np
from ultralytics import YOLO


class ObjectDetector:
    def __init__(self, model_path: str = "yolov8n.pt", target_classes: list = None):
        print(f"[ObjectDetector] Loading YOLO model: {model_path} ...")
        self.model = YOLO(model_path)   # auto-downloads on first run
        self.target_classes = target_classes or []
        print(f"[ObjectDetector] Tracking classes: {self.target_classes}")
        print("[ObjectDetector] Ready.")

    def detect(self, frame: np.ndarray) -> list:
        """
        Run YOLO detection on a BGR frame.
        Returns list of dicts:
            {
                'label': str,
                'confidence': float,
                'bbox': (x1, y1, x2, y2)   # pixel coords
            }
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            label = self.model.names[int(box.cls)]
            if self.target_classes and label not in self.target_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])

            detections.append({
                "label":      label,
                "confidence": round(conf, 2),
                "bbox":       (x1, y1, x2, y2),
            })

        return detections
