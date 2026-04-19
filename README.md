# BlindSpot AI — Driver Assistance System
### Final Year Project | Computer Vision | Deep Learning

---

## Overview
Real-time blind spot and obstacle detection system for driver assistance.
Combines **MiDaS depth estimation** and **YOLOv8 object detection** to identify
nearby vehicles, pedestrians, and road hazards — with audio alerts and a live dashboard.

---

## Features
- 🎯 Real-time object detection (cars, people, trucks, motorcycles, etc.)
- 📏 Distance estimation in metres using depth map + bounding box fusion
- 🔊 Audio alerts — double beep on DANGER, single beep on WARNING
- 🖥️ Live dashboard UI with FPS counter, alert banner, detection list
- 🗺️ Depth map inset (toggle with D key)
- 📝 CSV logging of all detections for analysis

---

## Alert Zones
| Zone    | Distance  | Colour | Audio         |
|---------|-----------|--------|---------------|
| DANGER  | ≤ 2.0 m   | Red    | Double beep   |
| WARNING | ≤ 4.0 m   | Orange | Single beep   |
| SAFE    | > 4.0 m   | Green  | None          |

---

## Project Structure
```
BlindSpotDetection/
├── main.py                  ← Entry point
├── requirements.txt
├── core/
│   ├── depth_estimator.py   ← MiDaS depth estimation
│   ├── object_detector.py   ← YOLOv8 detection
│   ├── distance_calculator.py ← Metric distance fusion
│   └── alert_manager.py     ← Alert logic + audio
├── ui/
│   └── dashboard.py         ← Real-time UI rendering
├── utils/
│   └── logger.py            ← CSV detection logger
└── output/                  ← Screenshots + logs saved here
```

---

## Installation

**Step 1 — Install Python 3.10+** from https://python.org

**Step 2 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 3 — Run the system:**
```bash
python main.py
```

---

## Controls
| Key | Action             |
|-----|--------------------|
| ESC or Q | Quit          |
| S   | Save screenshot    |
| D   | Toggle depth map   |

---

## Configuration
Edit the `CONFIG` dictionary in `main.py` to tune:
- `danger_distance_m` — danger threshold (default: 2.0m)
- `warning_distance_m` — warning threshold (default: 4.0m)
- `camera_index` — change if using external webcam (try 1, 2)
- `enable_audio` — turn alerts on/off
- `yolo_model` — swap to `yolov8s.pt` for more accuracy

---

## Technologies Used
| Technology | Purpose |
|------------|---------|
| Python 3.10 | Core language |
| PyTorch | Deep learning framework |
| MiDaS (Intel) | Monocular depth estimation |
| YOLOv8 (Ultralytics) | Real-time object detection |
| OpenCV | Video capture & UI rendering |

---

## Author
**Saurabh Kalugade**
sourabhkalugade479@gmail.com
