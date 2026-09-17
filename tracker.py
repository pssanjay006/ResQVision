#!/usr/bin/env python
"""
tracker.py - ResQVision MVP, STEP 2b: detection + ByteTrack tracking.

Wraps Ultralytics' built-in tracker so the rest of the project never talks to
the Ultralytics API directly:

    tracker = VehicleTracker(model, device="0", half=True)
    analysis = tracker.analyze(frame)
    if analysis.accident_detected:
        ...   # analysis.vehicle_count, analysis.confidence, analysis.detections

One `model.track(...)` call per frame yields BOTH the accident boxes and the
vehicle boxes with track IDs, so detection and tracking share a single forward
pass. The tracker tracks every class we hand it; only vehicle-class IDs are
treated as vehicle IDs, and accident boxes keep their IDs only for debugging.

Standalone run, saves an annotated video with track IDs plus the per-frame
accident-hit sequence that the confirmer will consume in step 3:

    python tracker.py --source data/demo_clip.mp4
    python tracker.py --source data/demo_clip.mp4 --max-frames 200
    python tracker.py --source data/jam_clip.mp4
    python tracker.py --source data/demo_clip.mp4 --tracker botsort.yaml
    python tracker.py --source 0 --max-frames 200            # webcam

Output goes to data/annotated_tracked_<source>.mp4 by default.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from detector import (
    ACCIDENT_CLASS,
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_IOU_GATE,
    DEFAULT_RESIZE_WIDTH,
    DEFAULT_STRIDE,
    BASE_DIR,
    Detection,
    FrameStream,
    VideoWriterLazy,
    class_name_map,
    draw_detections,
    frame_has_accident,
    frame_has_accident_any,
    load_model,
    max_accident_confidence,
    max_accident_vehicle_iou,
    resolve_device,
    results_to_detections,
)

DEFAULT_TRACKER = "bytetrack.yaml"

# The model's NMS IoU is a different knob from the accident<->vehicle overlap
# gate. Keep them separate so tuning one cannot silently loosen the other.
DEFAULT_NMS_IOU = 0.5


@dataclass(frozen=True)
class FrameAnalysis:
    """Everything the pipeline needs to know about one processed frame."""

    detections: list[Detection]
    accident_detected: bool
    vehicle_count: int
    confidence: float            # accident confidence, overlap-gated
    raw_accident_confidence: float  # highest accident conf, gate ignored
    unique_vehicle_ids: int
    inference_ms: float

    @property
    def accident_boxes(self) -> list[Detection]:
        return [d for d in self.detections if d.cls == ACCIDENT_CLASS]

    @property
    def vehicle_boxes(self) -> list[Detection]:
        return [d for d in self.detections if d.cls != ACCIDENT_CLASS]


class VehicleTracker:
    """Thin, stateful wrapper over Ultralytics tracking.

    `persist=True` keeps the tracker's internal Kalman/association state alive
    between calls, which is what gives stable IDs across frames. `reset()` drops
    that state, so a new clip never inherits IDs from the previous one.
    """

    def __init__(
        self,
        model,
        tracker: str = DEFAULT_TRACKER,
        conf: float = DEFAULT_CONF,
        nms_iou: float = DEFAULT_NMS_IOU,
        imgsz: int = DEFAULT_IMGSZ,
        device: str = "0",
        half: bool = True,
        gate_iou: float = DEFAULT_IOU_GATE,
        require_vehicle_overlap: bool = True,
    ):
        self.model = model
        self.tracker = tracker
        self.conf = conf
        self.nms_iou = nms_iou
        self.imgsz = imgsz
        self.device = device
        self.half = half
        self.gate_iou = gate_iou
        # True  = rule A, the specified behaviour: accident box must overlap a
        #         vehicle box. Measured to block every frame of demo_clip, because
        #         the model's `vehicle` class fires in only 14% of its frames.
        # False = rule B, frame_has_accident_any(): any accident box counts.
        self.require_vehicle_overlap = require_vehicle_overlap
        self.names = {int(k): str(v) for k, v in model.names.items()}
        self.frame_count = 0

    def update(self, frame) -> list[Detection]:
        """Run detection + tracking on one frame; return Detection objects."""
        results = self.model.track(
            source=frame,
            tracker=self.tracker,
            persist=True,
            conf=self.conf,
            iou=self.nms_iou,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        self.frame_count += 1
        return results_to_detections(results[0], self.names)

    def analyze(self, frame) -> FrameAnalysis:
        """Run one frame and reduce it to the fields the pipeline consumes."""
        tick = time.perf_counter()
        detections = self.update(frame)
        inference_ms = (time.perf_counter() - tick) * 1000.0

        if self.require_vehicle_overlap:
            hit, vehicle_count, confidence = frame_has_accident(
                detections, iou_threshold=self.gate_iou
            )
        else:
            hit, vehicle_count, confidence = frame_has_accident_any(detections)
        return FrameAnalysis(
            detections=detections,
            accident_detected=hit,
            vehicle_count=vehicle_count,
            confidence=confidence,
            raw_accident_confidence=max_accident_confidence(detections),
            unique_vehicle_ids=len(unique_vehicle_ids(detections)),
            inference_ms=inference_ms,
        )

    def reset(self) -> None:
        """Drop tracker state. Setting predictor to None forces re-init on the
        next call, which is the reliable way to clear the built-in trackers."""
        self.model.predictor = None
        self.frame_count = 0


def unique_vehicle_ids(detections: Sequence[Detection]) -> set[int]:
    """Distinct vehicle track IDs currently in frame (accident boxes excluded)."""
    return {
        d.track_id
        for d in detections
        if d.cls != ACCIDENT_CLASS and d.track_id is not None
    }


def hit_sequence_line(hits: Sequence[bool], width: int = 80) -> str:
    """ASCII timeline of accident hits, for eyeballing before tuning step 3."""
    return "".join("1" if hit else "0" for hit in hits[:width]) + (
        "..." if len(hits) > width else ""
    )


# --------------------------------------------------------------------------- #
# Standalone runner: detection + tracking
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2b: run detection + ByteTrack over one video and save an annotated copy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="video path, image path, or webcam index (e.g. 0)")
    parser.add_argument("--weights", default=None, help="local .pt or an Ultralytics name (e.g. yolo11s.pt)")
    parser.add_argument("--output", default=None, help="output video path (default data/annotated_tracked_<source>.mp4)")
    parser.add_argument("--tracker", default=DEFAULT_TRACKER, help="bytetrack.yaml or botsort.yaml")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--nms-iou", type=float, default=DEFAULT_NMS_IOU, help="model NMS IoU (default 0.5)")
    parser.add_argument("--gate-iou", type=float, default=DEFAULT_IOU_GATE, help="accident<->vehicle gate IoU (default 0.3)")
    parser.add_argument("--no-vehicle-gate", action="store_true",
                        help="rule B: accept any accident box, drop the vehicle-overlap requirement "
                             "(needed to reach CONFIRMED on data/demo_clip.mp4)")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="process every Nth frame (default 2)")
    parser.add_argument("--resize-width", type=int, default=DEFAULT_RESIZE_WIDTH)
    parser.add_argument("--device", default=None, help="'0' for GPU, 'cpu' to force CPU")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after N processed frames")
    parser.add_argument("--no-half", action="store_true", help="force FP32 even on GPU")
    return parser.parse_args()


def default_output_path(source) -> Path:
    stem = Path(str(source)).stem if not str(source).isdigit() else "webcam"
    return BASE_DIR / "data" / f"annotated_tracked_{stem}.mp4"


def main() -> None:
    args = parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        sys.exit("[error] torch is missing. Run: pip install -r requirements.txt (see the CUDA note at the top)")

    device = resolve_device(args.device)
    half = (device != "cpu") and not args.no_half
    print(f"[device] device='{device}' half={half} tracker={args.tracker}")

    model = load_model(args.weights)
    print(f"[model] classes: {class_name_map(model)}")

    stream = FrameStream(args.source, stride=args.stride, resize_width=args.resize_width)
    print(
        f"[input] {stream.source}  source={stream.source_size[0]}x{stream.source_size[1]} "
        f"@ {stream.fps:.1f} fps, {stream.total_frames or 'unknown'} frames"
    )
    print(f"[input] resized to {stream.output_size[0]}x{stream.output_size[1]}, stride {stream.stride}")

    tracker = VehicleTracker(
        model,
        tracker=args.tracker,
        conf=args.conf,
        nms_iou=args.nms_iou,
        imgsz=args.imgsz,
        device=device,
        half=half,
        gate_iou=args.gate_iou,
        require_vehicle_overlap=not args.no_vehicle_gate,
    )

    if args.no_vehicle_gate:
        print("[gate] rule B: any accident box counts (vehicle-overlap gate disabled)")
    else:
        print(f"[gate] rule A: accident box must overlap a vehicle box at IoU >= {args.gate_iou}")

    writer = VideoWriterLazy(args.output or default_output_path(args.source), stream.output_fps)

    frames_processed = 0
    hits = 0
    hit_flags: list[bool] = []
    all_vehicle_ids: set[int] = set()
    ids_per_frame: list[int] = []
    best_gate_iou = 0.0
    total_inference_ms = 0.0
    started = time.perf_counter()

    try:
        for frame_index, frame in stream.frames():
            analysis = tracker.analyze(frame)
            frames_processed += 1
            total_inference_ms += analysis.inference_ms

            present_ids = unique_vehicle_ids(analysis.detections)
            all_vehicle_ids |= present_ids
            ids_per_frame.append(len(present_ids))
            best_gate_iou = max(best_gate_iou, max_accident_vehicle_iou(analysis.detections))
            hit_flags.append(analysis.accident_detected)
            if analysis.accident_detected:
                hits += 1

            elapsed = time.perf_counter() - started
            hud = [
                f"frame {frame_index} | {frames_processed} processed | tracker {args.tracker}",
                f"vehicles: {analysis.vehicle_count} (ids now {len(present_ids)}, ids total {len(all_vehicle_ids)})"
                f" | accident boxes: {len(analysis.accident_boxes)}",
                f"accident conf gated {analysis.confidence:.2f} / raw {analysis.raw_accident_confidence:.2f}"
                f" | gate IoU>={args.gate_iou}: {'HIT' if analysis.accident_detected else 'miss'}",
                f"{analysis.inference_ms:.0f} ms/frame | {frames_processed / elapsed:.1f} FPS wall",
            ]
            writer.write(
                draw_detections(
                    frame,
                    analysis.detections,
                    hud,
                    gate_hit=analysis.accident_detected,
                )
            )

            if frames_processed % 25 == 0:
                print(
                    f"  ... {frames_processed} frames | {total_inference_ms / frames_processed:.0f} ms/frame "
                    f"| {hits} hit frames | {len(all_vehicle_ids)} vehicle IDs"
                )
    except KeyboardInterrupt:
        print("\n[stop] interrupted")
    finally:
        writer.release()
        stream.release()

    if frames_processed == 0:
        sys.exit("[error] no frames processed")

    wall = time.perf_counter() - started
    print("\n" + "=" * 72)
    print("TRACKING SUMMARY")
    print(f"  frames processed          : {frames_processed}")
    print(f"  accident-hit frames       : {hits} ({100.0 * hits / frames_processed:.1f}%)")
    print(f"  distinct vehicle IDs seen : {len(all_vehicle_ids)}")
    print(f"  mean IDs visible / frame  : {sum(ids_per_frame) / len(ids_per_frame):.1f}")
    print(f"  best accident<->vehicle IoU: {best_gate_iou:.3f}  (gate is {args.gate_iou})")
    print(f"  mean inference            : {total_inference_ms / frames_processed:.1f} ms/frame")
    print(f"  wall-clock throughput     : {frames_processed / wall:.1f} processed FPS")
    print(f"  hit sequence (first 80)   : {hit_sequence_line(hit_flags)}")
    print(f"  annotated video           : {writer.path}")
    print("=" * 72)
    if hits:
        runs = longest_true_run(hit_flags)
        print(
            f"[your confirmer] window_size=10 / hit_ratio=0.8 needs 8 of 10 consecutive\n"
            f"                 processed frames to be 1. Longest unbroken run here: {runs}."
        )
    else:
        print("[your confirmer] zero hit frames, so this clip can never reach CONFIRMED (expected for a jam).")
        if best_gate_iou > 0.0:
            print(
                f"[your confirmer] but accident boxes WERE present - the gate blocked them "
                f"(best IoU {best_gate_iou:.3f} < {args.gate_iou}).\n"
                f"                 Re-run with --gate-iou {max(0.01, round(best_gate_iou * 0.5, 3))} "
                f"before concluding the clip is clean."
            )
    print(
        f"\nNOTE: with stride {stream.stride} at {stream.fps:.0f} fps the tracker sees "
        f"{stream.fps / stream.stride:.1f} fps, and 10 processed frames = "
        f"{10 * stream.stride / stream.fps:.2f} s of real time."
    )
    print(
        "\nREPORT BACK: mean ms/frame, wall FPS, distinct vehicle IDs, mean IDs/frame,\n"
        "the hit sequence line, and whether ID labels look stable in the output video."
    )


def longest_true_run(flags: Sequence[bool]) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


if __name__ == "__main__":
    main()
