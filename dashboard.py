"""
ui/dashboard.py
Renders the complete real-time UI dashboard on the video frame.

Layout:
┌─────────────────────────────────────────┐
│  [BlindSpot AI]          FPS: xx        │
│                                         │
│   [Camera feed with bounding boxes]     │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │  DANGER ZONE  /  WARNING  / SAFE │   │  ← Alert banner
│  └──────────────────────────────────┘   │
│  Objects: car 1.2m  person 3.4m  ...   │  ← Detection list
└─────────────────────────────────────────┘
"""

import cv2
import numpy as np


# Colour palette (BGR)
COLOURS = {
    "DANGER":   (0,   0,   220),   # red
    "WARNING":  (0,   165, 255),   # orange
    "SAFE":     (0,   200, 80),    # green
    "box_bg":   (20,  20,  20),
    "text":     (240, 240, 240),
    "accent":   (255, 200, 0),
    "dim":      (120, 120, 120),
}

LABEL_COLOURS = {
    "person":       (255, 100, 100),
    "car":          (100, 200, 255),
    "truck":        (100, 255, 200),
    "bus":          (200, 100, 255),
    "motorcycle":   (255, 200, 100),
    "bicycle":      (150, 255, 150),
}


class Dashboard:
    def __init__(self, width: int = 640, height: int = 480):
        self.w = width
        self.h = height

    def render(self, frame, detections, depth_colormap, alert_level, fps, frame_count):
        """Composite all UI elements onto the frame and return the result."""
        canvas = frame.copy()

        # ── Bounding boxes + labels ──────────────────────────────────────────
        for det in detections:
            self._draw_detection(canvas, det, alert_level)

        # ── Depth map overlay (top-right corner) ────────────────────────────
        if depth_colormap is not None:
            self._draw_depth_inset(canvas, depth_colormap)

        # ── Top bar ──────────────────────────────────────────────────────────
        self._draw_top_bar(canvas, fps, frame_count)

        # ── Alert banner ─────────────────────────────────────────────────────
        self._draw_alert_banner(canvas, alert_level, detections)

        # ── Detection list panel ─────────────────────────────────────────────
        self._draw_detection_list(canvas, detections)

        return canvas

    # ── Private helpers ──────────────────────────────────────────────────────

    def _draw_detection(self, canvas, det, alert_level):
        x1, y1, x2, y2 = det["bbox"]
        label    = det["label"]
        dist     = det.get("distance_m", 0)
        conf     = det.get("confidence", 0)

        # Box colour based on distance
        if dist <= 2.0:
            colour = COLOURS["DANGER"]
        elif dist <= 4.0:
            colour = COLOURS["WARNING"]
        else:
            colour = LABEL_COLOURS.get(label, COLOURS["SAFE"])

        thickness = 3 if dist <= 2.0 else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)

        # Corner accents
        clen = 15
        for px, py, dx, dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(canvas, (px, py), (px + dx*clen, py), colour, 3)
            cv2.line(canvas, (px, py), (px, py + dy*clen), colour, 3)

        # Label pill
        tag = f"{label}  {dist:.1f}m  {int(conf*100)}%"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        pill_y = max(y1 - 24, 0)
        cv2.rectangle(canvas, (x1, pill_y), (x1 + tw + 10, pill_y + th + 8), colour, -1)
        cv2.putText(canvas, tag, (x1 + 5, pill_y + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 10, 10), 1, cv2.LINE_AA)

    def _draw_depth_inset(self, canvas, depth_colormap):
        inset_w, inset_h = 200, 150
        inset = cv2.resize(depth_colormap, (inset_w, inset_h))
        x_off = self.w - inset_w - 10
        y_off = 50
        canvas[y_off:y_off+inset_h, x_off:x_off+inset_w] = inset
        cv2.rectangle(canvas, (x_off-1, y_off-1),
                      (x_off+inset_w+1, y_off+inset_h+1), COLOURS["dim"], 1)
        cv2.putText(canvas, "DEPTH MAP", (x_off+4, y_off+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1, cv2.LINE_AA)

    def _draw_top_bar(self, canvas, fps, frame_count):
        cv2.rectangle(canvas, (0, 0), (self.w, 38), (15, 15, 15), -1)
        cv2.putText(canvas, "BlindSpot AI", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOURS["accent"], 2, cv2.LINE_AA)
        fps_text = f"FPS: {fps:.1f}   Frame: {frame_count}"
        cv2.putText(canvas, fps_text, (self.w - 220, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOURS["text"], 1, cv2.LINE_AA)

    def _draw_alert_banner(self, canvas, alert_level, detections):
        colour = COLOURS[alert_level]
        y_start = self.h - 90

        # Semi-transparent banner background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, y_start), (self.w, y_start + 36), colour, -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        messages = {
            "DANGER":  "⚠  DANGER — OBJECT IN BLIND SPOT!",
            "WARNING": "!  WARNING — APPROACHING OBJECT",
            "SAFE":    "✓  ALL CLEAR — ROAD SAFE",
        }
        cv2.putText(canvas, messages[alert_level],
                    (self.w // 2 - 160, y_start + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2, cv2.LINE_AA)

    def _draw_detection_list(self, canvas, detections):
        y_base = self.h - 50
        cv2.rectangle(canvas, (0, y_base), (self.w, self.h), (15, 15, 15), -1)

        if not detections:
            cv2.putText(canvas, "No objects detected", (10, y_base + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOURS["dim"], 1, cv2.LINE_AA)
            return

        # Sort by distance ascending
        sorted_dets = sorted(detections, key=lambda d: d.get("distance_m", 99))
        x = 10
        for det in sorted_dets[:6]:   # max 6 items
            dist = det.get("distance_m", 0)
            col = COLOURS["DANGER"] if dist <= 2.0 else \
                  COLOURS["WARNING"] if dist <= 4.0 else COLOURS["SAFE"]
            tag = f"{det['label']} {dist:.1f}m"
            cv2.putText(canvas, tag, (x, y_base + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            x += tw + 20
            if x > self.w - 100:
                break
