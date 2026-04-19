"""
core/depth_estimator.py
Wraps the Intel MiDaS model for real-time monocular depth estimation.
"""

import cv2
import torch
import numpy as np


class DepthEstimator:
    def __init__(self, model_type: str = "DPT_Hybrid"):
        print(f"[DepthEstimator] Loading MiDaS model: {model_type} ...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DepthEstimator] Using device: {self.device}")

        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type in ("DPT_Large", "DPT_Hybrid"):
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

        print("[DepthEstimator] Ready.")

    def estimate(self, frame: np.ndarray):
        """
        Run depth estimation on a BGR frame.
        Returns:
            depth_map      - raw float32 depth array (higher = farther in MiDaS)
            depth_colormap - uint8 BGR colorised depth for display
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy().astype(np.float32)

        # Normalise to 0-255 for visualisation
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

        return depth_map, depth_colormap
