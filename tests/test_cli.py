import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from deepfake_scan.cli import main
from deepfake_scan.video import find_videos


def _write_video(path: Path, brightness: list, size=(64, 48)):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, size
    )
    assert writer.isOpened(), f"cannot write test video: {path}"
    for value in brightness:
        writer.write(np.full((size[1], size[0], 3), value, np.uint8))
    writer.release()
    assert path.is_file()


def test_find_videos(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    _write_video(tmp_path / "clip.avi", [10, 20, 30])
    _write_video(tmp_path / "sub" / "other.avi", [10, 20, 30])
    (tmp_path / "notes.txt").write_text("not a video")
    videos = find_videos(tmp_path)
    assert len(videos) == 2
    assert all(v.suffix == ".avi" for v in videos)


def test_cli_writes_report(tmp_path: Path, capsys):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    _write_video(video_dir / "smooth.avi", [i * 8 for i in range(12)])
    report_path = tmp_path / "report.json"

    code = main(
        [
            str(video_dir),
            "--threshold",
            "0.5",
            "--report",
            str(report_path),
            "--heatmaps",
            str(tmp_path / "heatmaps"),
        ]
    )

    assert code == 0
    report = json.loads(report_path.read_text())
    assert report["summary"]["total"] == 1
    result = report["results"][0]
    assert result["verdict"] == "real"
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["frames_analyzed"] > 0
    assert len(result["heatmaps"]) == 1
    assert Path(result["heatmaps"][0]).is_file()
    assert "scanned 1 video" in capsys.readouterr().out


def test_threshold_tunes_verdict(tmp_path: Path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    # Alternating still/moving segments maximize inter-frame irregularity.
    brightness = [0, 0, 220, 220, 0, 0, 220, 220, 0, 0, 220, 220]
    _write_video(video_dir / "jittery.avi", brightness)

    for threshold, verdict in (("0.5", "deepfake"), ("0.95", "real")):
        report_path = tmp_path / f"report-{threshold}.json"
        code = main(
            [
                str(video_dir),
                "--threshold",
                threshold,
                "--report",
                str(report_path),
                "--no-heatmaps",
            ]
        )
        assert code == 0
        report = json.loads(report_path.read_text())
        assert report["results"][0]["verdict"] == verdict


def test_fail_on_deepfake_exit_code(tmp_path: Path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    _write_video(
        video_dir / "jittery.avi", [0, 0, 220, 220, 0, 0, 220, 220]
    )

    code = main(
        [
            str(video_dir),
            "--threshold",
            "0.5",
            "--report",
            str(tmp_path / "r.json"),
            "--no-heatmaps",
            "--fail-on-deepfake",
        ]
    )
    assert code == 1

    code = main(
        [
            str(video_dir),
            "--threshold",
            "0.95",
            "--report",
            str(tmp_path / "r2.json"),
            "--no-heatmaps",
            "--fail-on-deepfake",
        ]
    )
    assert code == 0


def test_empty_directory(tmp_path: Path):
    empty = tmp_path / "videos"
    empty.mkdir()
    report_path = tmp_path / "report.json"
    code = main(
        [
            str(empty),
            "--report",
            str(report_path),
            "--no-heatmaps",
        ]
    )
    assert code == 0
    report = json.loads(report_path.read_text())
    assert report["summary"]["total"] == 0


def test_invalid_threshold(tmp_path: Path):
    with pytest.raises(SystemExit):
        main([str(tmp_path), "--threshold", "1.5"])


def test_missing_directory(tmp_path: Path):
    with pytest.raises(SystemExit):
        main([str(tmp_path / "nope")])
