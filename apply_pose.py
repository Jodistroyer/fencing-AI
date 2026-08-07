"""
Drop a fencing (or any) video into input_videos/, then run:

    python apply_pose.py

Output with skeleton overlay is written to output_videos/.
Uses your NVIDIA GPU automatically when CUDA PyTorch is installed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def resolve_device(requested: str | None) -> str:
    if requested is not None:
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except ImportError:
        pass
    return "cpu"


def collect_videos(input_dir: Path, explicit: list[str] | None) -> list[Path]:
    if explicit:
        videos = []
        for item in explicit:
            path = Path(item)
            if not path.is_file():
                raise FileNotFoundError(f"Video not found: {path}")
            videos.append(path)
        return videos

    # Skip helper clips we generate ourselves
    videos = sorted(
        p for p in input_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTS
        and not p.name.startswith("_preview")
    )
    return videos


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Apply pose estimation to videos dropped in input_videos/"
    )
    parser.add_argument(
        "videos",
        nargs="*",
        help="Optional video paths. If omitted, processes everything in input_videos/",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=root / "input_videos",
        help="Folder to watch for dropped videos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output_videos",
        help="Folder for pose-overlay results",
    )
    parser.add_argument(
        "--model",
        default="yolov8n-pose.pt",
        help="Ultralytics pose model (n/s/m/l/x). Larger = slower/more accurate.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cpu, 0 (first GPU), or leave blank for auto",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference size (lower = faster)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold",
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use FP16 on GPU (faster). Default: on for GPU, off for CPU.",
    )
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "Missing dependency. Install with:\n"
            "  pip install -r requirements-pose.txt",
            file=sys.stderr,
        )
        return 1

    args.input_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        videos = collect_videos(args.input_dir, args.videos or None)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not videos:
        print(
            f"No videos found.\n"
            f"Drop an .mp4 (or similar) into:\n  {args.input_dir}\n"
            f"Then run this script again."
        )
        return 0

    device = resolve_device(args.device)
    use_half = args.half if args.half is not None else (device != "cpu")

    print(f"Loading model: {args.model}")
    print(f"Device: {device}  |  half={use_half}")
    model = YOLO(args.model)

    predict_kwargs = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "device": device,
        "save": True,
        "project": str(args.output_dir.resolve()),
        "name": "runs",
        "exist_ok": True,
        "stream": True,
        "verbose": False,
    }
    # FP16 speeds up GPU inference on modern NVIDIA cards
    if use_half:
        predict_kwargs["half"] = True

    for video in videos:
        print(f"Processing: {video.name}")
        started = time.perf_counter()
        results = model.predict(source=str(video), **predict_kwargs)
        frame_count = 0
        for _ in results:
            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.perf_counter() - started
                fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"  ... {frame_count} frames ({fps:.1f} fps)")
        elapsed = time.perf_counter() - started
        fps = frame_count / elapsed if elapsed > 0 else 0
        print(f"  done ({frame_count} frames in {elapsed:.1f}s, {fps:.1f} fps)")

    out_run = args.output_dir / "runs"
    print(f"\nDone. Pose videos are in:\n  {out_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
