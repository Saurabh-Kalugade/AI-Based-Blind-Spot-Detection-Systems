"""
core/distance_calculator.py
Estimates real-world distance (metres) to each detected object
by sampling the MiDaS depth map inside the bounding box.

MiDaS returns *inverse* relative depth (higher value = closer).
We convert to approximate metric distance using a calibration scale.
"""

import numpy as np


class DistanceCalculator:
    def __init__(self, focal_length_px: float = 600, known_width_m: float = 0.45):
        """
        focal_length_px  : camera focal length in pixels (calibrate with a ruler)
        known_width_m    : reference object width in metres (default: shoulder width)
        """
        self.focal_length = focal_length_px
        self.known_width  = known_width_m

        # Scale factor to convert MiDaS relative depth → approximate metres
        # Tune this after calibration with a ruler test
        self.depth_scale  = 10.0

    def calculate(self, detections: list, depth_map: np.ndarray, frame_shape: tuple) -> list:
        """
        For each detection, sample the median depth inside its bounding box
        and estimate distance in metres.

        Returns extended detections list with added keys:
            'distance_m'    : float — estimated metres to object
            'zone'          : str   — 'DANGER' | 'WARNING' | 'SAFE'
            'depth_sample'  : float — raw MiDaS depth value
        """
        h, w = frame_shape[:2]
        results = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]

            # Clamp to frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            # Sample depth region (centre 50% of bbox to avoid edges)
            cx1 = x1 + (x2 - x1) // 4
            cy1 = y1 + (y2 - y1) // 4
            cx2 = x2 - (x2 - x1) // 4
            cy2 = y2 - (y2 - y1) // 4

            roi = depth_map[cy1:cy2, cx1:cx2]
            if roi.size == 0:
                continue

            # MiDaS: higher value = closer. Take median for robustness.
            depth_val = float(np.median(roi))

            # Convert to approximate metres (inverse relationship)
            # distance ≈ scale / depth_value
            if depth_val > 0:
                distance_m = round(self.depth_scale / depth_val * 255, 2)
            else:
                distance_m = 99.0   # unknown

            # Pixel width → metric distance (secondary estimate for validation)
            pixel_width = x2 - x1
            if pixel_width > 0:
                distance_bbox = round((self.known_width * self.focal_length) / pixel_width, 2)
            else:
                distance_bbox = distance_m

            # Blend both estimates (depth map is primary)
            final_distance = round(0.6 * distance_m + 0.4 * distance_bbox, 2)
            final_distance = max(0.1, min(final_distance, 50.0))   # clamp 0.1–50 m

            results.append({
                **det,
                "distance_m":   final_distance,
                "depth_sample": round(depth_val, 2),
            })

        return results
