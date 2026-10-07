"""EfficientNet-B4 + LSTM deepfake detection pipeline with temporal Grad-CAM.

Architecture (matches the project README):
    frames -> EfficientNet-B4 (shared weights) -> per-frame features
           -> LSTM (temporal consistency) -> classifier (Real/Fake)

Explainability: the fake-class score is backpropagated through the LSTM to
every frame's final convolutional feature map (temporal Grad-CAM).
"""

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

FRAME_SIZE = 380
DEFAULT_NUM_FRAMES = 16
DEFAULT_CLIP_SIZE = 16
DEFAULT_MAX_CLIPS = 2
DEFAULT_KEYFRAMES = 4
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FEATURE_DIM = 1792  # EfficientNet-B4 final conv channels


class TemporalDeepfakeModel(nn.Module):
    """EfficientNet-B4 spatial backbone + LSTM temporal model."""

    def __init__(self, num_classes=2, lstm_hidden=256, lstm_layers=2, dropout=0.3):
        super().__init__()
        self.backbone = torchvision.models.efficientnet_b4(weights=None)
        self.lstm = nn.LSTM(
            FEATURE_DIM, lstm_hidden, lstm_layers, batch_first=True, dropout=dropout
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(lstm_hidden, num_classes)
        )

    def extract(self, frames):
        """(B, 3, H, W) -> (B, FEATURE_DIM)."""
        features = self.backbone.features(frames)
        features = nn.functional.adaptive_avg_pool2d(features, 1)
        return torch.flatten(features, 1)

    def forward(self, frames):
        """(T, 3, H, W) single clip -> (1, num_classes) logits."""
        features = self.extract(frames)
        lstm_out, _ = self.lstm(features.unsqueeze(0))
        return self.classifier(lstm_out[:, -1])


class TemporalGradCAM:
    """Grad-CAM with gradient flow through the LSTM to each frame's conv map."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer or model.backbone.features[-1]
        self.activations = []
        self.handle = self.target_layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if torch.is_grad_enabled():
            output.retain_grad()
            self.activations.append(output)

    def heatmaps(self, frames_tensor, class_idx=1):
        """Return one Grad-CAM heatmap per frame in the clip.

        frames_tensor: (T, 3, H, W) preprocessed clip tensor.
        """
        self.model.eval()
        self.activations.clear()
        logits = self.model(frames_tensor)
        score = logits[0, class_idx]
        self.model.zero_grad()
        score.backward()
        maps = []
        for activation in self.activations:
            # activation: (T, C, H, W) — batched frames from one backbone pass
            grads = activation.grad
            weights = grads.mean(dim=(2, 3))  # (T, C)
            cams = torch.relu(
                (weights[:, :, None, None] * activation.detach()).sum(dim=1)
            )  # (T, H, W)
            for cam in cams:
                cam = cam - cam.min()
                if cam.max() > 0:
                    cam = cam / cam.max()
                maps.append(cam.cpu().numpy().astype(np.float32))
        return maps

    def close(self):
        self.handle.remove()


def sample_frames(video_path, num_frames=DEFAULT_NUM_FRAMES, frame_size=FRAME_SIZE):
    """Uniformly sample num_frames RGB frames from a video.

    Returns (num_frames, frame_size, frame_size, 3) uint8 array.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError("Video has no decodable frames")
    indices = np.linspace(0, total - 1, num_frames).astype(int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            cap.release()
            raise ValueError(f"Could not decode frame {idx}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_AREA)
        frames.append(frame)
    cap.release()
    return np.stack(frames).astype(np.uint8)


def preprocess(frames):
    """(T, H, W, 3) uint8 RGB -> (T, 3, H, W) normalized float tensor."""
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std


def overlay_heatmap(frame, heatmap, alpha=0.5):
    """Blend a jet-colormap heatmap onto an RGB frame."""
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)


def _hstack(images, gap=8):
    height = images[0].shape[0]
    separator = np.full((height, gap, 3), 255, np.uint8)
    rows = [images[0]]
    for image in images[1:]:
        rows.extend([separator, image])
    return np.concatenate(rows, axis=1)


def side_by_side(frame, heatmap_overlay):
    """One original frame next to its heatmap overlay."""
    return _hstack([frame, heatmap_overlay])


def heatmap_strip(pairs, cols=4):
    """Grid of side-by-side pairs, up to `cols` pairs per row."""
    rows = [pairs[i : i + cols] for i in range(0, len(pairs), cols)]
    row_images = [_hstack(row) for row in rows]
    if len(row_images) == 1:
        return row_images[0]
    width = max(image.shape[1] for image in row_images)
    padded = []
    for image in row_images:
        if image.shape[1] < width:
            pad = np.full((image.shape[0], width - image.shape[1], 3), 255, np.uint8)
            image = np.concatenate([image, pad], axis=1)
        padded.append(image)
    separator = np.full((8, width, 3), 255, np.uint8)
    stacked = [padded[0]]
    for image in padded[1:]:
        stacked.extend([separator, image])
    return np.concatenate(stacked, axis=0)


class DeepfakePipeline:
    """End-to-end pipeline: frame sampling -> per-clip prediction -> Grad-CAM."""

    def __init__(self, weights_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TemporalDeepfakeModel().to(self.device)
        self.model.eval()
        self.weights_source = "none"
        if weights_path:
            self.load_weights(weights_path)
        self.gradcam = TemporalGradCAM(self.model)

    def load_weights(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.weights_source = os.path.basename(path)

    @property
    def demo_mode(self):
        return self.weights_source == "none"

    def _predict_clip(self, frames):
        tensor = preprocess(frames).to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.model(tensor), dim=1)[0]
        return float(probs[0]), float(probs[1])

    def analyze(self, frames, clip_size=DEFAULT_CLIP_SIZE, max_clips=DEFAULT_MAX_CLIPS,
                num_keyframes=DEFAULT_KEYFRAMES):
        """Analyze sampled frames.

        frames: (N, H, W, 3) uint8 RGB from sample_frames().
        Returns a dict with per-clip results, overall verdict, and a
        side-by-side heatmap strip for the most suspicious clip.
        """
        if frames.ndim != 4 or frames.shape[0] == 0:
            raise ValueError("No frames to analyze")
        total_frames = frames.shape[0]
        clips = [
            frames[i : i + clip_size]
            for i in range(0, total_frames, clip_size)
        ][:max_clips]

        clip_results = []
        for idx, clip in enumerate(clips):
            p_real, p_fake = self._predict_clip(clip)
            clip_results.append(
                {
                    "clip": idx + 1,
                    "frames": f"{idx * clip_size + 1}-{idx * clip_size + len(clip)}",
                    "p_real": p_real,
                    "p_fake": p_fake,
                    "verdict": "FAKE" if p_fake > p_real else "REAL",
                }
            )

        overall_p_fake = float(np.mean([c["p_fake"] for c in clip_results]))
        overall_p_real = 1.0 - overall_p_fake

        # Grad-CAM on the most suspicious clip
        target_idx = int(np.argmax([c["p_fake"] for c in clip_results]))
        target_clip = clips[target_idx]
        tensor = preprocess(target_clip).to(self.device)
        heatmaps = self.gradcam.heatmaps(tensor, class_idx=1)

        keyframe_idx = np.linspace(
            0, len(heatmaps) - 1, min(num_keyframes, len(heatmaps))
        ).astype(int)
        pairs = []
        for idx in keyframe_idx:
            frame = target_clip[idx]
            overlay = overlay_heatmap(frame, heatmaps[idx])
            pairs.append((frame, overlay))

        strip = heatmap_strip([side_by_side(f, o) for f, o in pairs])
        gallery = [side_by_side(f, o) for f, o in pairs]

        return {
            "verdict": "FAKE" if overall_p_fake > overall_p_real else "REAL",
            "p_real": overall_p_real,
            "p_fake": overall_p_fake,
            "clips": clip_results,
            "keyframe_clip": target_idx + 1,
            "strip": strip,
            "gallery": gallery,
        }


def resolve_weights(weights_arg=None):
    """Locate model weights: CLI arg, local weights/best.pth, or HF Hub."""
    candidates = []
    if weights_arg:
        candidates.append(weights_arg)
    candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "weights", "best.pth"))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    hub_ref = os.environ.get("HF_WEIGHTS", "Lubnaaziz-28/deepfake-detection-demo")
    if "/" in hub_ref:
        repo_id, filename = hub_ref.split("/", 1)
        try:
            from huggingface_hub import hf_hub_download

            return hf_hub_download(repo_id=repo_id, filename=filename)
        except Exception:
            return None
    return None
