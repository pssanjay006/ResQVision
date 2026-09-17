#!/usr/bin/env python
"""
detector.py - ResQVision MVP, STEP 2a: detection only (no tracking).

Owns:
  * weight resolution + model loading
  * frame reading (OpenCV), resize-to-width, frame skipping
  * Ultralytics Results -> Detection objects
  * the xyxy IoU helper and the accident<->vehicle overlap gate
  * bounding-box / HUD drawing
  * a standalone annotated-video runner (__main__)

Standalone run, saves an annotated video so you can eyeball detection quality
BEFORE any tracking or UI work:

    python detector.py --source data/demo_clip.mp4
    python detector.py --source data/demo_clip.mp4 --max-frames 150
    python detector.py --source data/jam_clip.mp4 --conf 0.25
    python detector.py --source data/demo_clip.mp4 --weights yolo11s.pt
    python detector.py --source 0 --max-frames 200          # webcam
    python detector.py --source demo.mp4 --imgsz 480 --stride 3

Output goes to data/annotated_detected_<source>.mp4 by default.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

ACCIDENT_CLASS = "accident"
VEHICLE_CLASS = "vehicle"

BASE_DIR = Path(__file__).resolve().parent
MODEL_REPO = "Enos-123/traffic-accident-detection-yolo11x"
MODEL_FILENAME = "weights/epoch61.pt"

DEFAULT_RESIZE_WIDTH = 480
DEFAULT_STRIDE = 2
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 640
DEFAULT_IOU_GATE = 0.3

WEBCAM_DEFAULT_MAX_FRAMES = 300


# --------------------------------------------------------------------------- #
# Detection container
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Detection:
    """One detected box. `box` is (x1, y1, x2, y2) in pixel coordinates."""

    cls: str
    confidence: float
    box: tuple[float, float, float, float]
    track_id: int | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        return max(0.0, self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)


# --------------------------------------------------------------------------- #
# Device / weights
# --------------------------------------------------------------------------- #


def resolve_device(requested: str | None) -> str:
    """Return the Ultralytics device string, falling back to CPU without CUDA."""
    import torch

    if requested == "cpu":
        return "cpu"
    if requested is None:
        return "0" if torch.cuda.is_available() else "cpu"
    return requested


def resolve_weights(weights: str | None = None) -> str:
    """Resolve a local .pt, an Ultralytics model name, or download from HF."""
    if weights:
        if Path(weights).exists():
            return str(weights)
        return weights  # assume an official name such as yolo11s.pt

    cached = BASE_DIR / MODEL_FILENAME
    if cached.exists():
        return str(cached)

    from huggingface_hub import hf_hub_download

    print(f"[weights] downloading {MODEL_REPO}/{MODEL_FILENAME} (~114 MB, first run only)")
    return str(
        hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME, local_dir=str(BASE_DIR))
    )


def load_model(weights: str | None = None, fuse: bool = True):
    """Load the YOLO model. Imported lazily so the logic below stays testable."""
    from ultralytics import YOLO

    resolved = resolve_weights(weights)
    print(f"[model] loading {resolved}")
    model = YOLO(resolved)
    if fuse:
        try:
            model.fuse()
        except Exception as exc:  # noqa: BLE001 - fuse() is optional
            print(f"[model] fuse() skipped: {exc}")
    return model


def class_name_map(model) -> dict[str, int]:
    """Return {'accident': 0, 'vehicle': 1, ...} from the model's own names."""
    return {str(name): int(idx) for idx, name in model.names.items()}


# --------------------------------------------------------------------------- #
# Results -> Detection
# --------------------------------------------------------------------------- #


def to_detections(boxes, names: dict[int, str]) -> list[Detection]:
    """Convert an Ultralytics Boxes object into a list of Detection."""
    detections: list[Detection] = []
    if boxes is None:
        return detections

    ids = boxes.id
    for index, box in enumerate(boxes):
        cls_id = int(box.cls.item())
        coords = tuple(float(v) for v in box.xyxy[0].tolist())
        track_id = None
        if ids is not None:
            track_id = int(ids[index].item())
        detections.append(
            Detection(
                cls=names.get(cls_id, str(cls_id)),
                confidence=float(box.conf.item()),
                box=coords,  # type: ignore[arg-type]
                track_id=track_id,
            )
        )
    return detections


def results_to_detections(result, names: dict[int, str]) -> list[Detection]:
    """Convert a single Ultralytics Result (one frame) into Detections."""
    return to_detections(getattr(result, "boxes", None), names)


# --------------------------------------------------------------------------- #
# IoU + the overlap gate  (the exact logic specified for this project)
# --------------------------------------------------------------------------- #


def iou(box_a, box_b) -> float:
    """Standard intersection-over-union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = box_a[0], box_a[1], box_a[2], box_a[3]
    bx1, by1, bx2, by2 = box_b[0], box_b[1], box_b[2], box_b[3]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def frame_has_accident(detections, iou_threshold=0.3):
    """Accident box overlapping a vehicle box counts as an accident frame.

    Returns (hit, vehicle_count, best_accident_confidence).

    NOTE: best_conf only considers accident boxes that actually overlap a
    vehicle. If a large accident box overlaps several vehicles this is the
    highest of those; if nothing overlaps it stays 0.0 even when the detector
    did fire.
    """
    accident_boxes = [d for d in detections if d.cls == "accident"]
    vehicle_boxes = [d for d in detections if d.cls == "vehicle"]
    best_conf = 0.0
    hit = False
    for a in accident_boxes:
        for v in vehicle_boxes:
            if iou(a.box, v.box) >= iou_threshold:
                hit = True
                best_conf = max(best_conf, a.confidence)
    return hit, len(vehicle_boxes), best_conf


def frame_has_accident_any(detections):
    """RULE B: any accident box is a hit; the vehicle-overlap requirement is dropped.

    Same return contract as frame_has_accident():
    (hit, vehicle_count, best_accident_confidence).

    Why this exists - measured on the two demo clips with these exact settings
    (YOLO11x weights/epoch61.pt, conf 0.25, imgsz 640, half, stride 2, 480px wide):

        data/demo_clip.mp4   accident box in  89/115 frames (77%), max conf 0.96
                             VEHICLE box in only 16/115 frames (14%), 0.2/frame
            rule A -> 5 hits, longest run 2      -> does NOT confirm  (FAILS the test)
            rule B -> 89 hits                    -> CONFIRMS at processed frame 33
        data/jam_clip.mp4    accident box in    3/370 frames ( 1%), max conf 0.42
                             VEHICLE box in  370/370 frames (100%), 21.6/frame
            rule A -> 3 hits, longest run 1      -> does not confirm  (correct)
            rule B -> 3 hits, longest run 3      -> does not confirm  (correct)

    The gate blocks the demo clip because the model's `vehicle` class barely
    fires there (0.2 boxes/frame), so there is usually nothing to overlap with.
    Dropping the detector confidence to 0.05 does NOT change this: the same 16
    frames have vehicle boxes. This is a model/domain issue, not a threshold one.

    jam_clip stays safe under rule B by a wide margin (needs 8 of 10 consecutive
    hits, longest run is 3), so rule B is not simply "confirm everything".
    """
    accident_boxes = [d for d in detections if d.cls == ACCIDENT_CLASS]
    vehicle_boxes = [d for d in detections if d.cls == VEHICLE_CLASS]
    best_conf = max((d.confidence for d in accident_boxes), default=0.0)
    return bool(accident_boxes), len(vehicle_boxes), best_conf


def max_accident_confidence(detections) -> float:
    """Highest accident confidence in the frame, regardless of vehicle overlap.

    Diagnostic only: if this is high while frame_has_accident() returns False,
    the IoU gate is what is blocking you, not the detector.
    """
    confidences = [d.confidence for d in detections if d.cls == ACCIDENT_CLASS]
    return max(confidences) if confidences else 0.0


def max_accident_vehicle_iou(detections) -> float:
    """Highest accident<->vehicle IoU in the frame (0.0 if either is absent).

    Diagnostic only. Run a clip, read this number off the summary, and set
    --gate-iou just below it. Measured examples:
      * accident box tightly inside one vehicle box  -> 0.04
      * accident box wrapping the whole scene        -> 0.05
      * accident box roughly overlaying a vehicle    -> 0.33+
    """
    accident_boxes = [d for d in detections if d.cls == ACCIDENT_CLASS]
    vehicle_boxes = [d for d in detections if d.cls == VEHICLE_CLASS]
    best = 0.0
    for a in accident_boxes:
        for v in vehicle_boxes:
            best = max(best, iou(a.box, v.box))
    return best


# --------------------------------------------------------------------------- #
# Frame input
# --------------------------------------------------------------------------- #


def normalize_source(source):
    """Video/image path -> str, webcam index -> int."""
    if isinstance(source, int):
        return source
    path = Path(str(source))
    if path.exists():
        return str(path)
    text = str(source)
    if text.isdigit():
        return int(text)
    raise FileNotFoundError(f"source not found: {source}")


class FrameStream:
    """Reads a video/webcam/image, resizing every frame to `resize_width`.

    Iterating yields (original_frame_index, resized_bgr_frame) for every
    `stride`-th frame only. Skipped frames are still read, because a capture
    must decode them to advance.
    """

    def __init__(self, source, stride: int = DEFAULT_STRIDE, resize_width: int = DEFAULT_RESIZE_WIDTH):
        self.source = normalize_source(source)
        self.stride = max(1, int(stride))
        self.resize_width = max(64, int(resize_width))
        self.is_webcam = isinstance(self.source, int)

        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise IOError(f"OpenCV could not open source: {source}")

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not self.fps or self.fps != self.fps:  # 0 or NaN
            self.fps = 25.0
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        source_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH) or self.resize_width)
        source_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or self.resize_width)
        self.source_size = (source_width, source_height)
        self.resize_height = int(round(source_height * self.resize_width / max(1, source_width)))
        self.output_size = (self.resize_width, self.resize_height)
        # Processed frames are written back at fps / stride so the saved video
        # still plays at real-time speed.
        self.output_fps = max(1.0, self.fps / self.stride)

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        index = -1
        while True:
            ok, frame = self.capture.read()
            if not ok:
                return
            index += 1
            if index % self.stride != 0:
                continue
            yield index, cv2.resize(
                frame, self.output_size, interpolation=cv2.INTER_LINEAR
            )

    def release(self) -> None:
        self.capture.release()

    def __enter__(self) -> "FrameStream":
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()


# --------------------------------------------------------------------------- #
# Frame output
# --------------------------------------------------------------------------- #


class VideoWriterLazy:
    """Opens the writer on the first frame, once the frame size is known."""

    def __init__(self, path, fps: float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.writer = None
        self.fourcc_name = None

    def _candidates(self) -> tuple[str, ...]:
        if self.path.suffix.lower() == ".avi":
            return ("XVID", "MJPG")
        return ("mp4v", "avc1", "MJPG")

    def write(self, frame: np.ndarray) -> None:
        if self.writer is None:
            height, width = frame.shape[:2]
            for name in self._candidates():
                writer = cv2.VideoWriter(
                    str(self.path),
                    cv2.VideoWriter_fourcc(*name),
                    self.fps,
                    (width, height),
                )
                if writer.isOpened():
                    self.writer = writer
                    self.fourcc_name = name
                    print(f"[output] writing {self.path} ({width}x{height} @ {self.fps:.1f} fps, {name})")
                    break
                writer.release()
            if self.writer is None:
                raise RuntimeError(
                    f"no working video codec for {self.path}; try --output with a .avi suffix"
                )
        self.writer.write(frame)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #

ACCIDENT_COLOR = (0, 0, 255)      # red
ACCIDENT_MATCHED_COLOR = (0, 0, 255)
ACCIDENT_UNMATCHED_COLOR = (0, 165, 255)  # orange
VEHICLE_COLOR = (255, 176, 0)     # blue-ish
TEXT_COLOR = (255, 255, 255)


def draw_detections(
    frame: np.ndarray,
    detections: Sequence[Detection],
    hud_lines: Sequence[str] = (),
    show_track_ids: bool = True,
    gate_hit: bool = False,
) -> np.ndarray:
    """Draw accident/vehicle boxes, optional track IDs, and a HUD bar."""
    for detection in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in detection.box)
        if detection.cls == ACCIDENT_CLASS:
            color = ACCIDENT_MATCHED_COLOR if gate_hit else ACCIDENT_UNMATCHED_COLOR
            thickness = 3 if gate_hit else 2
            label = f"{ACCIDENT_CLASS} {detection.confidence:.2f}"
        else:
            color = VEHICLE_COLOR
            thickness = 1
            if show_track_ids and detection.track_id is not None:
                label = f"{detection.cls}#{detection.track_id} {detection.confidence:.2f}"
            else:
                label = f"{detection.cls} {detection.confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - text_h - 4)), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, max(text_h, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 0) if color == VEHICLE_COLOR else (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if hud_lines:
        lines = [line for line in hud_lines if line]
        bar_height = 14 * len(lines) + 8
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        for row, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (6, 15 + row * 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 220, 255) if "HIT" in line else TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
    return frame


# --------------------------------------------------------------------------- #
# Standalone runner: detection only
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2a: run accident/vehicle detection over one video and save an annotated copy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="video path, image path, or webcam index (e.g. 0)")
    parser.add_argument("--weights", default=None, help="local .pt or an Ultralytics name (e.g. yolo11s.pt)")
    parser.add_argument("--output", default=None, help="output video path (default data/annotated_detected_<source>.mp4)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_GATE, help="accident<->vehicle gate IoU (default 0.3)")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="process every Nth frame (default 2)")
    parser.add_argument("--resize-width", type=int, default=DEFAULT_RESIZE_WIDTH)
    parser.add_argument("--device", default=None, help="'0' for GPU, 'cpu' to force CPU")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after N processed frames")
    parser.add_argument("--no-half", action="store_true", help="force FP32 even on GPU")
    return parser.parse_args()


def default_output_path(source) -> Path:
    stem = Path(str(source)).stem if not str(source).isdigit() else "webcam"
    return BASE_DIR / "data" / f"annotated_detected_{stem}.mp4"


def main() -> None:
    args = parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("[error] torch is missing. Run: pip install -r requirements.txt (see the CUDA note at the top)")

    device = resolve_device(args.device)
    half = (device != "cpu") and not args.no_half
    print(f"[device] device='{device}' half={half}")

    model = load_model(args.weights)
    names = {int(k): str(v) for k, v in model.names.items()}
    print(f"[model] classes: {names}")
    if ACCIDENT_CLASS not in names.values():
        print(f"[warn] no '{ACCIDENT_CLASS}' class in this model - the overlap gate will never fire")
    if VEHICLE_CLASS not in names.values():
        print(f"[warn] no '{VEHICLE_CLASS}' class in this model - vehicle tracking will have nothing to track")

    stream = FrameStream(args.source, stride=args.stride, resize_width=args.resize_width)
    max_frames = args.max_frames
    if stream.is_webcam and max_frames is None:
        max_frames = WEBCAM_DEFAULT_MAX_FRAMES
        print(f"[input] webcam detected, stopping after {max_frames} processed frames")

    print(
        f"[input] {stream.source}  source={stream.source_size[0]}x{stream.source_size[1]} "
        f"@ {stream.fps:.1f} fps, {stream.total_frames or 'unknown'} frames"
    )
    print(f"[input] resized to {stream.output_size[0]}x{stream.output_size[1]}, stride {stream.stride}")

    writer = VideoWriterLazy(args.output or default_output_path(args.source), stream.output_fps)

    frames_processed = 0
    frames_with_accident_box = 0
    frames_with_gate_hit = 0
    boxes_by_class: dict[str, int] = {}
    vehicle_counts: list[int] = []
    accident_confs: list[tuple[float, int]] = []
    best_gate_iou = 0.0
    total_inference_ms = 0.0
    started = time.perf_counter()

    try:
        for frame_index, frame in stream.frames():
            tick = time.perf_counter()
            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=device,
                half=half,
                verbose=False,
            )
            inference_ms = (time.perf_counter() - tick) * 1000.0
            total_inference_ms += inference_ms

            detections = results_to_detections(results[0], names)
            hit, vehicle_count, best_conf = frame_has_accident(detections, iou_threshold=args.iou)
            raw_conf = max_accident_confidence(detections)
            best_gate_iou = max(best_gate_iou, max_accident_vehicle_iou(detections))

            for detection in detections:
                boxes_by_class[detection.cls] = boxes_by_class.get(detection.cls, 0) + 1
            accident_count = boxes_by_class.get(ACCIDENT_CLASS, 0)

            if raw_conf > 0.0:
                frames_with_accident_box += 1
                accident_confs.append((raw_conf, frame_index))
            if hit:
                frames_with_gate_hit += 1
            vehicle_counts.append(vehicle_count)

            frames_processed += 1
            elapsed = time.perf_counter() - started
            hud = [
                f"frame {frame_index} | {frames_processed} processed | {args.imgsz}px conf>={args.conf}",
                f"accident boxes: {sum(1 for d in detections if d.cls == ACCIDENT_CLASS)} "
                f"(max conf {raw_conf:.2f}) | vehicles: {vehicle_count}",
                f"gate IoU>={args.iou}: {'HIT' if hit else 'miss'}"
                f"{f' at conf {best_conf:.2f}' if hit else ''}",
                f"{inference_ms:.0f} ms/frame | {frames_processed / elapsed:.1f} FPS wall",
            ]
            writer.write(draw_detections(frame, detections, hud, gate_hit=hit))

            if frames_processed % 25 == 0:
                print(
                    f"  ... {frames_processed} frames | {total_inference_ms / frames_processed:.0f} ms/frame "
                    f"| {boxes_by_class.get(ACCIDENT_CLASS, 0)} accident boxes so far"
                )

            if max_frames and frames_processed >= max_frames:
                break
    except KeyboardInterrupt:
        print("\n[stop] interrupted")
    finally:
        writer.release()
        stream.release()

    if frames_processed == 0:
        sys.exit("[error] no frames processed")

    wall = time.perf_counter() - started
    print("\n" + "=" * 72)
    print("DETECTION SUMMARY")
    print(f"  frames processed            : {frames_processed}")
    print(f"  boxes per class             : {boxes_by_class or '{}'}")
    print(f"  frames with an accident box : {frames_with_accident_box}")
    print(f"  frames passing the IoU gate : {frames_with_gate_hit}")
    print(f"  mean vehicle count / frame  : {sum(vehicle_counts) / len(vehicle_counts):.1f}")
    print(f"  best accident<->vehicle IoU : {best_gate_iou:.3f}  (gate is {args.iou})")
    print(f"  mean inference              : {total_inference_ms / frames_processed:.1f} ms/frame")
    print(f"  wall-clock throughput       : {frames_processed / wall:.1f} processed FPS")
    if accident_confs:
        top = sorted(accident_confs, reverse=True)[:5]
        print("  strongest accident frames   : " + ", ".join(f"#{idx}@{conf:.2f}" for conf, idx in top))
    print(f"  annotated video             : {writer.path}")
    print("=" * 72)
    if frames_with_accident_box and not frames_with_gate_hit:
        print(
            "[finding] the detector fired but the overlap gate NEVER passed.\n"
            f"          Highest accident<->vehicle IoU seen was {best_gate_iou:.3f}, below the "
            f"gate of {args.iou}.\n"
            "          IoU punishes size mismatch in BOTH directions: a small accident box\n"
            "          inside a big vehicle box scores ~0.04, and a scene-wide accident box\n"
            "          over small vehicle boxes scores ~0.05. Re-run with\n"
            f"          --iou {max(0.01, round(best_gate_iou * 0.5, 3))} to let those through."
        )
    if not frames_with_accident_box:
        print(
            "[finding] no accident boxes at all on this clip.\n"
            "          Try --conf 0.15, or use --weights yolo11s.pt, or this clip simply\n"
            "          has no accident-like frames (expected for jam_clip.mp4)."
        )
    print(
        "\nREPORT BACK: mean ms/frame, wall FPS, boxes-per-class counts, frames with an\n"
        "accident box, frames passing the IoU gate, and whether the output video looks right."
    )


if __name__ == "__main__":
    main()
