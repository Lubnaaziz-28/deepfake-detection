"""Video discovery and frame sampling utilities."""

from __future__ import annotations

import math
from pathlib import Path

import cv2

VIDEO_EXTENSIONS = frozenset(
    {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm",
        ".m4v",
        ".flv",
        ".wmv",
        ".mpg",
        ".mpeg",
        ".3gp",
    }
)


def find_videos(input_dir: Path) -> list[Path]:
    """Return every video file under input_dir (recursively), sorted."""
    if not input_dir.is_dir():
        return []
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def sample_frames(video_path: Path, num_frames: int = 16) -> list:
    """Sample num_frames evenly spaced from the video as RGB numpy arrays."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"cannot open video: {video_path}")
    try:
        raw_total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        total = int(raw_total) if not math.isnan(raw_total) and raw_total > 0 else 0
        if total > 0:
            step = max(num_frames - 1, 1)
            indices = sorted({round(i * (total - 1) / step) for i in range(num_frames)})
            frames = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if ok:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frames = []
            while len(frames) < num_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not frames:
            raise OSError(f"no decodable frames in video: {video_path}")
        return frames
    finally:
        capture.release()
