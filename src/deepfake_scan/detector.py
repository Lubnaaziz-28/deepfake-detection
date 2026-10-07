"""Detection engine: temporal-coherence baseline + optional trained model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .heatmap import overlay_heatmap, temporal_consistency_heatmap
from .video import find_videos, sample_frames


@dataclass
class ScanResult:
    path: str
    verdict: str  # "real" | "deepfake" | "error"
    confidence: float
    score: float
    frames_analyzed: int
    heatmaps: list = field(default_factory=list)
    error: str = ""


class DeepfakeDetector:
    """Score videos for deepfake likelihood.

    Without trained weights, a deterministic temporal-coherence
    heuristic (inter-frame motion irregularity) is used. With
    ``model_path`` pointing at trained weights, the EfficientNet-B4
    + LSTM model is used instead.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        frames_per_video: int = 16,
        heatmap_dir: Path | None = None,
        model_path: Path | None = None,
    ):
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0.0, 1.0]")
        self.threshold = threshold
        self.frames_per_video = frames_per_video
        self.heatmap_dir = heatmap_dir
        self.model_path = model_path
        self._model = _load_torch_model(model_path) if model_path else None

    def scan_directory(self, input_dir: Path) -> list[ScanResult]:
        """Scan every video under input_dir, returning one ScanResult each."""
        return [self.scan_file(path) for path in find_videos(input_dir)]

    def scan_file(self, path: Path) -> ScanResult:
        """Scan a single video file."""
        try:
            frames = sample_frames(path, self.frames_per_video)
        except Exception as exc:  # noqa: BLE001 - per-file errors are reported
            return ScanResult(str(path), "error", 0.0, 0.0, 0, [], str(exc))
        try:
            score = self._score(frames)
            heatmaps = self._write_heatmap(path, frames)
        except Exception as exc:  # noqa: BLE001 - per-file errors are reported
            return ScanResult(str(path), "error", 0.0, 0.0, len(frames), [], str(exc))
        verdict = "deepfake" if score >= self.threshold else "real"
        confidence = score if verdict == "deepfake" else 1.0 - score
        return ScanResult(
            str(path),
            verdict,
            round(float(confidence), 4),
            round(float(score), 4),
            len(frames),
            heatmaps,
            "",
        )

    def _score(self, frames: list) -> float:
        return self._model_score(frames) if self._model else self._temporal_score(frames)

    def _temporal_score(self, frames: list) -> float:
        """Temporal-coherence heuristic: irregular inter-frame motion scores high."""
        gray = [
            cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (64, 64)).astype(
                np.float32
            )
            for frame in frames
        ]
        if len(gray) < 2:
            return 0.0
        diffs = np.array([np.abs(gray[i + 1] - gray[i]).mean() for i in range(len(gray) - 1)])
        mean = diffs.mean()
        irregularity = float(np.abs(diffs - mean).mean() / (mean + 1e-6))
        flicker = min(1.0, 0.9 * irregularity)
        frozen = 1.0 if mean < 0.5 else 0.0
        return min(1.0, 0.85 * flicker + 0.15 * frozen)

    def _model_score(self, frames: list) -> float:
        import torch

        with torch.no_grad():
            logits = self._model(_stack_frames(frames))
        return float(torch.sigmoid(logits).item())

    def _write_heatmap(self, path: Path, frames: list) -> list:
        if self.heatmap_dir is None:
            return []
        if self._model is not None:
            try:
                heat_rgb = self._model.heatmap_overlay(frames)
            except Exception:  # noqa: BLE001 - fall back to temporal heatmap
                heat_rgb = temporal_consistency_heatmap(frames)
        else:
            heat_rgb = temporal_consistency_heatmap(frames)
        stem = "".join(c if c.isalnum() else "_" for c in path.stem)
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.heatmap_dir / f"{stem}_heatmap.png"
        from PIL import Image

        Image.fromarray(heat_rgb).save(out_path)
        return [str(out_path)]


def _stack_frames(frames: list):
    import torch
    from torchvision import transforms

    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return torch.stack([preprocess(frame) for frame in frames]).unsqueeze(0)


def _load_torch_model(model_path: Path):
    """Load the EfficientNet-B4 + LSTM detector from trained weights."""
    import torch
    from torch import nn
    from torchvision import models

    if not model_path.is_file():
        raise FileNotFoundError(f"model weights not found: {model_path}")

    class TemporalDetector(nn.Module):
        """EfficientNet-B4 spatial features + LSTM temporal head."""

        def __init__(self):
            super().__init__()
            self.backbone = models.efficientnet_b4(weights=None)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            self.lstm = nn.LSTM(in_features, 256, batch_first=True)
            self.head = nn.Linear(256, 1)

        def forward(self, x):
            time = x.shape[1]
            features = [self.backbone(x[:, t]) for t in range(time)]
            sequence = torch.stack(features, dim=1)
            output, _ = self.lstm(sequence)
            return self.head(output[:, -1]).squeeze(-1)

        def heatmap_overlay(self, frames: list):
            """Grad-CAM overlay of the final sampled frame's feature response."""
            import cv2
            import torch

            x = _stack_frames(frames)
            saved = {}

            def forward_hook(_module, _inputs, output):
                saved["acts"] = output

            handle = self.backbone.features.register_forward_hook(forward_hook)
            try:
                logits = self(x)
                activations = saved["acts"]
                gradients = torch.autograd.grad(logits[0], activations)[0]
            finally:
                handle.remove()
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activations).sum(dim=1)).squeeze(0)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            heat = cv2.resize(
                (cam * 255).byte().cpu().numpy(),
                (frames[0].shape[1], frames[0].shape[0]),
            )
            return overlay_heatmap(frames[len(frames) // 2], heat.astype(np.float32))

    model = TemporalDetector()
    checkpoint = torch.load(model_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model
