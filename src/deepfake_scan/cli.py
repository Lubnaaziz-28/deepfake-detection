"""deepfake-scan command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .detector import DeepfakeDetector
from .report import build_report, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deepfake-scan",
        description="Scan a directory of video files for deepfakes and emit a JSON report.",
    )
    parser.add_argument("input_dir", type=Path, help="directory to scan recursively")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="confidence threshold for a deepfake verdict, in (0.0, 1.0] (default: 0.5)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("deepfake-report.json"),
        help="path to write the JSON report (default: deepfake-report.json)",
    )
    parser.add_argument(
        "--heatmaps",
        type=Path,
        default=Path("deepfake-heatmaps"),
        help="directory to write heatmap overlays (default: deepfake-heatmaps)",
    )
    parser.add_argument(
        "--no-heatmaps",
        action="store_true",
        help="skip heatmap generation",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=16,
        help="frames sampled per video (default: 16)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="path to trained model weights (EfficientNet-B4 + LSTM checkpoint)",
    )
    parser.add_argument(
        "--fail-on-deepfake",
        action="store_true",
        help="exit with code 1 when any video is flagged as a deepfake",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"deepfake-scan {__version__}",
    )
    return parser


def main(argv: list | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not 0.0 < args.threshold <= 1.0:
        parser.error("--threshold must be in (0.0, 1.0]")
    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")
    if args.model is not None and not args.model.is_file():
        parser.error(f"model weights not found: {args.model}")
    if args.frames < 2:
        parser.error("--frames must be at least 2")

    heatmap_dir = None if args.no_heatmaps else args.heatmaps
    detector = DeepfakeDetector(
        threshold=args.threshold,
        frames_per_video=args.frames,
        heatmap_dir=heatmap_dir,
        model_path=args.model,
    )
    results = detector.scan_directory(args.input_dir)
    report = build_report(
        results,
        input_dir=args.input_dir,
        threshold=args.threshold,
        frames_per_video=args.frames,
        model_path=args.model,
    )
    write_report(report, args.report)

    summary = report["summary"]
    print(f"scanned {summary['total']} video(s) from {args.input_dir}")
    for result in results:
        if result.verdict == "error":
            print(f"  ERROR  {result.path}: {result.error}")
        else:
            print(
                f"  {result.verdict.upper():7} "
                f"confidence={result.confidence:.4f} "
                f"score={result.score:.4f} {result.path}"
            )
    print(f"report written to {args.report}")

    if args.fail_on_deepfake and summary["deepfakes"] > 0:
        print(
            f"deepfake(s) detected above threshold {args.threshold}: "
            + ", ".join(summary["flagged"])
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
