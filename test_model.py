#!/usr/bin/env python
"""
ResQVision MVP - STEP 1 sanity check.

Loads the fine-tuned accident detector
`Enos-123/traffic-accident-detection-yolo11x` (accident / vehicle) and runs ONE
image or video frame through it at confidence 0.25, 0.40 and 0.60, printing
every detected class, confidence and box. Also measures per-frame latency so we
can decide whether YOLO11x is usable on your machine or whether we fall back to
yolo11s.pt.

Run from inside resqvision-mvp/:

    python test_model.py                                  # auto-picks a sample
    python test_model.py --source data/demo_clip.mp4
    python test_model.py --source data/demo_clip.mp4 --video-time 3.5
    python test_model.py --device cpu
    python test_model.py --weights yolo11s.pt             # fallback detector

First run downloads ~114 MB of weights into ./weights/ and, if you have no clip
in data/, one of the model repo's own test images into ./testing/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_REPO = "Enos-123/traffic-accident-detection-yolo11x"
# epoch61.pt is the checkpoint whose metrics are reported on the model card
# (epoch14.pt is also published; swap --weights if you want to compare them).
MODEL_FILENAME = "weights/epoch61.pt"
SAMPLE_FILENAMES = ("testing/fig1.jpg", "testing/fig2.jpg", "testing/fig3.jpg")

CONFIDENCE_LEVELS = (0.25, 0.40, 0.60)
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 1: verify the YOLO11x accident detector runs on this machine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default=None,
        help="image or video file to test on. Auto-detected from data/ if omitted.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="local .pt path, or an Ultralytics model name to auto-download "
        "(e.g. yolo11s.pt). Defaults to the Hugging Face accident model.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="inference size (default 640)")
    parser.add_argument(
        "--device",
        default=None,
        help="'0' for the first CUDA GPU, 'cpu' to force CPU. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--video-time",
        type=float,
        default=1.0,
        help="seconds into a video to grab the test frame (default 1.0)",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=10,
        help="timed inference iterations after 2 warmup runs (default 10)",
    )
    return parser.parse_args()


def resolve_device(requested: str | None) -> str:
    """Return the Ultralytics device string, falling back to CPU when no CUDA."""
    import torch

    if requested == "cpu":
        return "cpu"
    if requested is None:
        return "0" if torch.cuda.is_available() else "cpu"
    return requested


def resolve_weights(weights_arg: str | None) -> str:
    """Return a path/name that YOLO() can load, downloading from HF if needed."""
    if weights_arg:
        path = Path(weights_arg)
        if path.exists():
            return str(path)
        # Not a local file: assume an official Ultralytics model name (yolo11s.pt).
        print(f"[weights] '{weights_arg}' is not a local file, letting Ultralytics resolve it")
        return weights_arg

    cached = BASE_DIR / MODEL_FILENAME
    if cached.exists():
        print(f"[weights] using cached {cached}")
        return str(cached)

    from huggingface_hub import hf_hub_download

    print(f"[weights] downloading {MODEL_REPO}/{MODEL_FILENAME} (~114 MB, first run only)")
    path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        local_dir=str(BASE_DIR),
    )
    print(f"[weights] saved to {path}")
    return str(path)


def resolve_source(source_arg: str | None) -> Path:
    """Return the image/video to test on, falling back to a bundled test image."""
    if source_arg:
        path = Path(source_arg)
        if not path.exists():
            sys.exit(f"[error] source not found: {path}")
        return path

    for candidate in (
        BASE_DIR / "data" / "demo_clip.mp4",
        BASE_DIR / "data" / "jam_clip.mp4",
        BASE_DIR / "data" / "sample.jpg",
        BASE_DIR / "data" / "sample.png",
    ):
        if candidate.exists():
            print(f"[source] using {candidate}")
            return candidate

    from huggingface_hub import hf_hub_download

    print("[source] nothing in data/ - falling back to a test image from the model repo")
    for filename in SAMPLE_FILENAMES:
        try:
            path = hf_hub_download(
                repo_id=MODEL_REPO,
                filename=filename,
                local_dir=str(BASE_DIR),
            )
            print(f"[source] using {path}")
            return Path(path)
        except Exception as exc:  # noqa: BLE001 - we just try the next candidate
            print(f"[source] could not fetch {filename}: {exc}")
    sys.exit("[error] no usable source. Pass --source <image-or-video>.")


def grab_frame(source: Path, video_time: float):
    """Return (frame_bgr, description). Videos yield a single frame."""
    import cv2

    if source.suffix.lower() in VIDEO_SUFFIXES:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            sys.exit(f"[error] OpenCV could not open video: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target = int(fps * video_time)
        if frame_count and target >= frame_count:
            target = frame_count // 2
        capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = capture.read()
        if not ok:  # seek failed - rewind and take the first frame
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        capture.release()
        if not ok:
            sys.exit(f"[error] could not read a frame from {source}")
        return frame, f"{source.name} frame {target} ({fps:.1f} fps, {frame.shape[1]}x{frame.shape[0]})"

    frame = cv2.imread(str(source))
    if frame is None:
        sys.exit(f"[error] OpenCV could not read image: {source}")
    return frame, f"{source.name} ({frame.shape[1]}x{frame.shape[0]})"


def print_report(model, frame, description: str, device: str, imgsz: int, half: bool) -> None:
    """Run the model at each confidence level and print what it found."""
    names = {int(k): str(v) for k, v in model.names.items()}
    print(f"\n[model] classes: {names}")
    if "accident" not in names.values():
        print("[model] WARNING: no 'accident' class - the pipeline's IoU gate will never fire")

    for conf in CONFIDENCE_LEVELS:
        start = time.perf_counter()
        results = model.predict(
            source=frame, conf=conf, imgsz=imgsz, device=device, half=half, verbose=False
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        boxes = results[0].boxes

        rows = []
        for box in boxes:
            cls_id = int(box.cls.item())
            rows.append(
                (
                    names.get(cls_id, str(cls_id)),
                    float(box.conf.item()),
                    [round(float(v), 1) for v in box.xyxy[0].tolist()],
                )
            )
        rows.sort(key=lambda row: -row[1])

        counts: dict[str, int] = {}
        for label, _, _ in rows:
            counts[label] = counts.get(label, 0) + 1

        print(f"\n--- conf >= {conf:.2f} --- {len(rows)} box(es)  [{elapsed_ms:.0f} ms incl. pre/post]")
        print(f"    counts: {counts or '{}'}")
        for label, score, box in rows[:12]:
            print(f"    {label:<10} {score:.3f}  xyxy={box}")
        if len(rows) > 12:
            print(f"    ... {len(rows) - 12} more")


def benchmark(model, frame, device: str, imgsz: int, half: bool, iterations: int) -> None:
    """Measure steady-state per-frame latency at the pipeline's default conf."""
    print(f"\n[bench] {iterations} timed runs at conf=0.25, imgsz={imgsz}, half={half}, device={device}")
    for _ in range(2):  # warmup - first call includes kernel autotuning
        model.predict(source=frame, conf=0.25, imgsz=imgsz, device=device, half=half, verbose=False)

    start = time.perf_counter()
    for _ in range(iterations):
        model.predict(source=frame, conf=0.25, imgsz=imgsz, device=device, half=half, verbose=False)
    per_frame_ms = (time.perf_counter() - start) * 1000.0 / iterations

    print(f"[bench] {per_frame_ms:.1f} ms/frame  ->  {1000.0 / per_frame_ms:.1f} FPS (detector only)")
    print("[bench] remember: tracking, the IoU gate and drawing will cut this roughly in half")


def print_device_info(device: str, half: bool) -> None:
    import torch

    print(f"[device] torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  using='{device}'  half={half}")
    if device != "cpu" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"[device] GPU: {props.name}  VRAM {props.total_memory / 1024**3:.1f} GB  sm_{props.major}{props.minor}")


def main() -> None:
    args = parse_args()

    try:
        import torch  # noqa: F401
        from ultralytics import YOLO
    except ImportError as exc:
        sys.exit(
            f"[error] missing dependency: {exc}\n"
            "        pip install -r requirements.txt\n"
            "        (GPU users: install the CUDA PyTorch wheel first - see requirements.txt)"
        )

    weights = resolve_weights(args.weights)
    device = resolve_device(args.device)
    half = device != "cpu"
    print_device_info(device, half)

    print(f"\n[model] loading {weights}")
    model = YOLO(weights)
    try:
        model.fuse()  # merges Conv+BN for a small, free speedup
    except Exception as exc:  # noqa: BLE001 - fuse() is a nicety, not required
        print(f"[model] fuse() skipped: {exc}")

    source = resolve_source(args.source)
    frame, description = grab_frame(source, args.video_time)
    print(f"\n[source] {description}")

    print_report(model, frame, description, device, args.imgsz, half)
    benchmark(model, frame, device, args.imgsz, half, args.bench_iters)

    print(
        "\n" + "=" * 72 + "\n"
        "REPORT BACK TO BUFFY:\n"
        "  1. Did the model download and load without errors?\n"
        "  2. The printed class list - is 'accident' present?\n"
        "  3. How many 'accident' boxes at conf 0.25 / 0.40 / 0.60 on your clip?\n"
        "  4. The ms/frame and FPS from [bench].\n"
        "  5. GPU name + VRAM (or 'CPU only').\n"
        + "=" * 72
    )


if __name__ == "__main__":
    main()
