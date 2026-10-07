"""Heatmap generation: temporal-consistency overlays and Grad-CAM blending."""

import cv2
import numpy as np


def overlay_heatmap(base_frame: np.ndarray, heat: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend a single-channel heat map (0-255) onto an RGB base frame."""
    base_bgr = cv2.cvtColor(base_frame, cv2.COLOR_RGB2BGR)
    heat_colored = cv2.applyColorMap(heat.astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(base_bgr, 1.0 - alpha, heat_colored, alpha, 0.0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def temporal_consistency_heatmap(frames: list) -> np.ndarray:
    """RGB overlay of average inter-frame motion, a proxy for temporal artifacts."""
    if not frames:
        raise ValueError("no frames provided")
    size = (128, 128)
    resized = [cv2.resize(frame, size) for frame in frames]
    if len(resized) < 2:
        return overlay_heatmap(resized[0], np.zeros(size[::-1], dtype=np.float32))
    diffs = [
        np.abs(resized[i + 1].astype(np.int16) - resized[i].astype(np.int16)).sum(axis=2)
        for i in range(len(resized) - 1)
    ]
    motion = np.mean(diffs, axis=0).astype(np.float32)
    motion = cv2.normalize(motion, None, 0, 255, cv2.NORM_MINMAX)
    base = resized[len(resized) // 2]
    return overlay_heatmap(base, motion)
