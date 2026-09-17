#!/usr/bin/env python
"""
app.py - ResQVision MVP, STEP 5: the Streamlit dashboard.

Ties the finished modules together:

    detector.VehicleTracker.analyze(frame) -> FrameAnalysis
        -> confirm.SimpleConfirmer.update_frame()   (multi-frame gate)
        -> severity.estimate_severity()             (rule-based label)
        -> location.get_nearest()                   (Overpass, once per incident)
        -> this file's panel                        (human confirmation)

Run:
    streamlit run app.py

THERE IS NO DATABASE AND NO REAL DISPATCH. Incidents live in
st.session_state and vanish when the tab closes. The "Confirm Alert" button
prints a simulated-notification message; it calls no API and contacts nobody.
The words SIMULATION appear on screen at all times, and must stay that way.

Scope reminders (deliberate, per the project brief):
    * one active incident at a time
    * one hardcoded demo camera location
    * no evidence clips, no face/plate handling
    * no multi-stage state machine - just the rolling window in confirm.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from detector import (
    BASE_DIR,
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    DEFAULT_IOU_GATE,
    DEFAULT_RESIZE_WIDTH,
    DEFAULT_STRIDE,
    MODEL_FILENAME,
    FrameStream,
    draw_detections,
    load_model,
    resolve_device,
    resolve_weights,
)
from confirm import SimpleConfirmer
from location import DEMO_CAMERA, FALLBACK_HOSPITAL, FALLBACK_POLICE, get_nearest
from severity import SEVERITY_NOTE, estimate_severity, severity_breakdown
from tracker import VehicleTracker

DATA_DIR = BASE_DIR / "data"

BANNER = "SIMULATION — no real emergency service is contacted by this system"
ALERT_SENT_MSG = "Simulated notification sent — no real service was contacted."
DISMISSED_MSG = "Incident dismissed as a false alarm. No notification was sent."
AWAITING_MSG = "Confirmed. A human must review this before anything is sent."

STATUS_MONITORING = "MONITORING"
STATUS_CONFIRMED = "CONFIRMED — awaiting human review"
STATUS_ALERT = "ALERT SIMULATED (human-confirmed)"
STATUS_DISMISSED = "DISMISSED — false alarm"

SEVERITY_COLORS = {"MINOR": "#2e7d32", "MODERATE": "#ef6c00", "SEVERE": "#c62828"}

DEVICE_NOTE = (
    "FP16 (half=True) only takes effect on a CUDA device. On CPU the run is FP32 "
    "and roughly 1-3 FPS with YOLO11x, which is why --stride and --max-frames exist."
)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

STATE_DEFAULTS = {
    "running": False,
    "incident": None,          # dict for the single active incident
    "review": None,            # None | "alert" | "false_alarm"
    "notice": None,            # transient message under the panel
    "confirmer": None,         # the live SimpleConfirmer
    "last_frame": None,        # last annotated frame, so the video survives a rerun
    "last_stats": None,        # dict from the most recent run
    "model_error": None,
}


def init_state() -> None:
    for key, value in STATE_DEFAULTS.items():
        st.session_state.setdefault(key, value)


def reset_incident(clear_notice: bool = True) -> None:
    """Drop the incident AND reset the confirmer.

    Resetting is mandatory on BOTH resolutions, not just the false alarm: once
    confirm.py latches `confirmed=True` it stays latched until reset(), and
    during its cooldown it ignores frames entirely. Forget this and the app goes
    permanently deaf after the first incident, however the human resolved it.
    """
    st.session_state.incident = None
    st.session_state.review = None
    if clear_notice:
        st.session_state.notice = None
    confirmer = st.session_state.get("confirmer")
    if confirmer is not None:
        confirmer.reset()


# --------------------------------------------------------------------------- #
# Model / source plumbing
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def get_model(weights: str):
    """Load YOLO once per process. Without cache_resource this reloads on every
    widget interaction, which on a 114 MB checkpoint is a multi-second stall."""
    return load_model(weights)


def available_sources() -> dict[str, str]:
    """Label -> source value, for whatever actually exists on disk."""
    options: dict[str, str] = {}
    for name in ("demo_clip.mp4", "jam_clip.mp4"):
        path = DATA_DIR / name
        if path.exists():
            label = f"data/{name}" + ("  (should CONFIRM)" if name.startswith("demo") else "  (should NOT confirm)")
            options[label] = str(path)
    options["Webcam (index 0)"] = "0"
    return options


def incident_status(incident: dict | None, review: str | None) -> str:
    if review == "alert":
        return STATUS_ALERT
    if review == "false_alarm":
        return STATUS_DISMISSED
    if incident is not None:
        return STATUS_CONFIRMED
    return STATUS_MONITORING


# --------------------------------------------------------------------------- #
# Incident construction
# --------------------------------------------------------------------------- #

def build_incident(analysis, frame_index: int, camera: dict, lookup_services: bool = True) -> dict:
    """Turn a confirmed frame into the incident record the panel renders.

    `lookup_services=False` skips the Overpass calls; the dashboard then shows
    the hardcoded fallbacks. Useful when demoing offline.
    """
    severity = estimate_severity(analysis.vehicle_count, analysis.confidence)
    incident = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "severity": severity,
        "breakdown": severity_breakdown(analysis.vehicle_count, analysis.confidence),
        "vehicle_count": analysis.vehicle_count,
        "unique_vehicle_ids": analysis.unique_vehicle_ids,
        "confidence": round(float(analysis.confidence), 3),
        "raw_confidence": round(float(analysis.raw_accident_confidence), 3),
        "frame_index": frame_index,
        "camera": camera,
    }

    if lookup_services:
        # Once per incident, never per frame - Overpass rate-limits hard.
        incident["hospital"] = get_nearest("hospital", camera["lat"], camera["lon"], FALLBACK_HOSPITAL)
        incident["police"] = get_nearest("police", camera["lat"], camera["lon"], FALLBACK_POLICE)
    else:
        incident["hospital"] = {**FALLBACK_HOSPITAL, "source": "fallback",
                                "distance_km": None, "error": "service lookup disabled"}
        incident["police"] = {**FALLBACK_POLICE, "source": "fallback",
                              "distance_km": None, "error": "service lookup disabled"}
    return incident


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def render_banner() -> None:
    st.markdown(
        f"<div style='background:#b71c1c;color:#fff;padding:0.5rem 0.75rem;"
        f"border-radius:0.375rem;font-weight:600;text-align:center;'>{BANNER}</div>",
        unsafe_allow_html=True,
    )


def render_confirmer_status() -> None:
    confirmer: SimpleConfirmer | None = st.session_state.get("confirmer")
    if confirmer is None:
        return
    status = confirmer.status()
    window = status["hit_sequence"] or "(empty)"
    if status["in_cooldown"]:
        st.caption(
            f"Confirmer: in cooldown {status['cooldown_remaining']:.0f}s — "
            f"frames are being ignored until the incident is resolved."
        )
    st.caption(
        f"Confirmer window: {status['hits']}/{status['window_frames']} hits "
        f"(confirms at {status['hit_ratio']:.0%} of {status['window_size']})  |  "
        f"`{window}`"
    )


def render_service(label: str, service: dict) -> None:
    source = service.get("source", "fallback")
    badge = "OpenStreetMap" if source == "overpass" else "hardcoded fallback"
    distance = service.get("distance_km")
    distance_text = f"{distance} km away" if distance is not None else "distance unavailable"
    st.markdown(f"**{label}:** {service['name']}")
    st.caption(f"{distance_text}  ·  source: {badge}")


def render_map(incident: dict) -> None:
    points = [
        {"lat": incident["camera"]["lat"], "lon": incident["camera"]["lon"],
         "label": "incident", "color": "#c62828"},
    ]
    for label, key, color in (("hospital", "hospital", "#2e7d32"), ("police", "police", "#1565c0")):
        service = incident.get(key)
        if service:
            points.append({"lat": service["lat"], "lon": service["lon"],
                           "label": label, "color": color})
    frame = pd.DataFrame(points)
    st.map(frame, latitude="lat", longitude="lon", color="color", size=60)
    st.caption("red = incident  ·  green = hospital  ·  blue = police  ·  map data © OpenStreetMap")


def render_panel() -> None:
    incident: dict | None = st.session_state.incident
    review: str | None = st.session_state.review
    status = incident_status(incident, review)

    st.subheader("Incident panel")
    if status == STATUS_ALERT:
        st.success(f"Status: {status}")
    elif status == STATUS_DISMISSED:
        st.info(f"Status: {status}")
    elif status == STATUS_CONFIRMED:
        st.warning(f"Status: {status}")
    else:
        st.info(f"Status: {status}")

    if incident is None:
        st.write("No incident. The camera is being monitored.")
        render_confirmer_status()
        if st.session_state.get("notice"):
            st.caption(st.session_state.notice)
        return

    if st.session_state.get("notice"):
        st.caption(st.session_state.notice)

    severity = incident["severity"]
    st.markdown(
        f"<div style='background:{SEVERITY_COLORS[severity]};color:#fff;padding:0.4rem 0.6rem;"
        f"border-radius:0.375rem;font-weight:600;'>Severity: {severity}</div>",
        unsafe_allow_html=True,
    )
    st.caption(SEVERITY_NOTE)

    left, right = st.columns(2)
    left.metric("Vehicles in frame", incident["vehicle_count"])
    right.metric("Accident confidence", f"{incident['confidence']:.2f}")

    breakdown = incident["breakdown"]
    st.caption(
        f"Heuristic score {breakdown['score']}/4 "
        f"(vehicles +{breakdown['vehicle_points']}, confidence +{breakdown['confidence_points']})"
    )

    st.markdown("---")
    camera = incident["camera"]
    st.markdown(f"**Location:** {camera['name']}  \n"
                f"`{camera['lat']:.4f}, {camera['lon']:.4f}` (fixed demo location)")
    st.caption(f"Detected at frame {incident['frame_index']} on {incident['created_at']}")

    st.markdown("**Nearest services**")
    render_service("Hospital", incident["hospital"])
    render_service("Police station", incident["police"])
    render_map(incident)

    st.markdown("---")
    if review is None:
        st.markdown("**Human confirmation required**")
        col_confirm, col_reject = st.columns(2)
        if col_confirm.button("Confirm Alert", type="primary", use_container_width=True,
                              key="btn_confirm_alert"):
            # NOTE: this also resets the confirmer. The brief only requires it on
            # "False Alarm", but without it here the app stays latched and deaf
            # after the first alert - the next incident would never be detected.
            st.session_state.review = "alert"
            st.session_state.notice = ALERT_SENT_MSG
            if st.session_state.get("confirmer") is not None:
                st.session_state.confirmer.reset()
            st.rerun()
        if col_reject.button("False Alarm", use_container_width=True, key="btn_false_alarm"):
            # reset_incident() also nulls `review`, so set it afterwards or the
            # dismissal state is wiped before it can render.
            reset_incident(clear_notice=False)
            st.session_state.review = "false_alarm"
            st.session_state.notice = DISMISSED_MSG
            st.rerun()
        st.caption(AWAITING_MSG)
    elif review == "alert":
        st.success(ALERT_SENT_MSG)
        if st.button("Clear incident", use_container_width=True, key="btn_clear_alert"):
            reset_incident()
            st.rerun()
    else:
        st.info(DISMISSED_MSG)
        if st.button("Clear incident", use_container_width=True, key="btn_clear_dismissed"):
            reset_incident()
            st.rerun()

    st.markdown("---")
    render_confirmer_status()


def render_video(frame, caption: str) -> None:
    if frame is None:
        st.info("No video yet — pick a source in the sidebar and press Start.")
        return
    st.image(frame, channels="BGR", use_container_width=True, caption=caption)


# --------------------------------------------------------------------------- #
# The run loop
# --------------------------------------------------------------------------- #

def run_stream(model, source, cfg: dict, video_slot, panel_slot) -> None:
    """Walk the source once, updating the video slot and the confirmer.

    Streamlit runs this inside a single script execution, so widgets clicked
    mid-loop are not handled until the loop exits. The loop therefore STOPPED
    the first time an incident is confirmed (unless you untick the box), which
    makes the Confirm/False Alarm buttons live as soon as there is something to
    review.
    """
    confirmer = SimpleConfirmer(
        window_size=cfg["window"], hit_ratio=cfg["ratio"], cooldown_sec=cfg["cooldown"]
    )
    st.session_state.confirmer = confirmer

    device = cfg["device"]
    half = cfg["half"] and device != "cpu"
    tracker = VehicleTracker(
        model,
        conf=cfg["conf"],
        imgsz=cfg["imgsz"],
        device=device,
        half=half,
        gate_iou=cfg["gate_iou"],
        require_vehicle_overlap=cfg["require_vehicle_overlap"],
    )

    processed = 0
    hit_frames = 0
    started = time.perf_counter()
    last_frame = None
    stop_reason = "source exhausted"
    min_frame_time = 1.0 / max(1.0, cfg["display_fps"])

    with FrameStream(source, stride=cfg["stride"], resize_width=cfg["resize_width"]) as stream:
        max_frames = cfg["max_frames"]
        if isinstance(stream.source, int) and not max_frames:
            # A webcam never "ends"; without a cap the loop would run forever and
            # the browser tab would never render another widget.
            max_frames = 300
            stop_reason = "webcam frame cap reached (set Max processed frames)"

        for frame_index, frame in stream.frames():
            tick = time.perf_counter()
            analysis = tracker.analyze(frame)
            if analysis.accident_detected:
                hit_frames += 1

            confirmed_now = confirmer.update_frame(analysis)
            status = confirmer.status()

            hud = [
                f"frame {frame_index}  stride {cfg['stride']}  {device}"
                f"{' fp16' if half else ''}  {analysis.inference_ms:.0f} ms",
                f"accident hit: {'YES' if analysis.accident_detected else 'no'}"
                f"  conf {analysis.confidence:.2f}"
                f"  raw {analysis.raw_accident_confidence:.2f}"
                f"  vehicles {analysis.vehicle_count}"
                f"  rule {'A (needs overlap)' if cfg['require_vehicle_overlap'] else 'B (accident box only)'}",
                f"window {status['hits']}/{status['window_frames']} "
                f"(need {status['hit_ratio']:.0%} of {status['window_size']})"
                f"{'  COOLDOWN' if status['in_cooldown'] else ''}",
            ]
            annotated = draw_detections(
                frame,
                analysis.detections,
                hud_lines=hud,
                show_track_ids=True,
                gate_hit=analysis.accident_detected,
            )
            last_frame = annotated
            video_slot.image(annotated, channels="BGR", use_container_width=True,
                             caption=f"processed frame {processed + 1}")

            if confirmed_now:
                with st.spinner("Accident confirmed — looking up nearest services "
                                "(Overpass, up to 2 x 5 s)..."):
                    incident = build_incident(
                        analysis, frame_index, DEMO_CAMERA,
                        lookup_services=cfg["lookup_services"],
                    )
                st.session_state.incident = incident
                st.session_state.review = None
                st.session_state.notice = None
                panel_slot.empty()
                if cfg["pause_on_confirm"]:
                    stop_reason = "paused: accident confirmed"
            else:
                panel_slot.caption(
                    f"Monitoring — window {status['hits']}/{status['window_frames']} hits, "
                    f"vehicles {analysis.vehicle_count}."
                )

            processed += 1
            elapsed = time.perf_counter() - tick
            if elapsed < min_frame_time:
                time.sleep(min_frame_time - elapsed)

            if confirmed_now and cfg["pause_on_confirm"]:
                break
            if max_frames and processed >= max_frames:
                stop_reason = f"reached the frame cap ({max_frames})"
                break

    duration = time.perf_counter() - started
    st.session_state.last_frame = last_frame
    st.session_state.last_stats = {
        "processed": processed,
        "hit_frames": hit_frames,
        "duration": duration,
        "fps": processed / duration if duration else 0.0,
        "stop_reason": stop_reason,
        "confirmed": st.session_state.incident is not None,
    }


# --------------------------------------------------------------------------- #
# Sidebar + main
# --------------------------------------------------------------------------- #

def render_sidebar() -> dict | None:
    st.sidebar.header("Source")
    sources = available_sources()
    upload = st.sidebar.file_uploader("Or upload a clip", type=["mp4", "avi", "mov", "mkv"])

    if upload is not None:
        upload_path = DATA_DIR / f"uploaded_{upload.name}"
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        upload_path.write_bytes(upload.getbuffer())
        source_options = {f"uploaded: {upload.name}": str(upload_path), **sources}
    else:
        source_options = sources

    if not source_options:
        st.sidebar.error("No clips found. Put demo_clip.mp4 / jam_clip.mp4 in data/ "
                         "or upload one.")
        return None

    label = st.sidebar.selectbox("Video source", list(source_options.keys()))
    source = source_options[label]

    st.sidebar.header("Pipeline")
    # Deliberately NOT resolve_weights(): that can download, and this runs on
    # every rerun. The heavier load happens once inside st.cache_resource.
    st.sidebar.caption(
        f"Model: {Path(MODEL_FILENAME).name} (classes: accident / vehicle). "
        "Detection + ByteTrack run in one pass per frame."
    )
    conf = st.sidebar.slider("Detector confidence", 0.05, 0.9, DEFAULT_CONF, 0.05,
                             help="Lower finds more accidents and more false positives. The "
                                  "confirmer, not this, is supposed to absorb the noise.")
    gate_iou = st.sidebar.slider("Accident<->vehicle IoU gate", 0.0, 0.5, DEFAULT_IOU_GATE, 0.01,
                                 help="MEASURED PROBLEM: realistic accident/vehicle box pairs "
                                      "score ~0.02-0.04 IoU, so 0.30 can block every frame. "
                                      "Use 0.01 unless you have measured otherwise.")
    require_vehicle_gate = st.sidebar.checkbox(
        "Require accident\u2194vehicle overlap (rule A)",
        value=True,
        help="MEASURED ON YOUR CLIPS: with this ticked nothing ever confirms. The model's "
             "VEHICLE class fires in only 16/115 frames of demo_clip.mp4 (0.2 boxes/frame), "
             "so the overlap gate blocks all but 5 frames and the window never fills. "
             "UNTICK THIS to reach CONFIRMED on demo_clip.mp4 (89 hits, confirms around "
             "processed frame 33). jam_clip.mp4 stays unconfirmed either way - its longest "
             "run of accident hits is 3, and the window needs 8 of 10.",
    )
    stride = st.sidebar.slider("Frame stride", 1, 5, DEFAULT_STRIDE, 1,
                               help="Process every Nth frame.")
    resize_width = st.sidebar.select_slider("Resize width", [320, 416, 480, 640],
                                            value=DEFAULT_RESIZE_WIDTH)
    imgsz = st.sidebar.select_slider("Detector imgsz", [320, 416, 480, 640], value=DEFAULT_IMGSZ,
                                     help="480 is cheaper than 480-then-upscale-to-640.")
    display_fps = st.sidebar.slider("Display FPS cap", 2, 15, 8, 1,
                                    help="Streamlit cannot do 30 FPS. 5-10 is the honest range.")
    max_frames = st.sidebar.number_input("Max processed frames", 0, 20000, 600, 50,
                                         help="0 = no limit.")
    device_choice = st.sidebar.selectbox("Device", ["auto", "0", "cpu"], index=0)

    with st.sidebar.expander("Confirmation window"):
        window = st.slider("Window size (frames)", 3, 60, 10, 1)
        ratio = st.slider("Hit ratio", 0.1, 1.0, 0.8, 0.05)
        cooldown = st.slider("Cooldown (s)", 0, 120, 30, 5)
        pause_on_confirm = st.checkbox("Pause on confirmation", value=True,
                                       help="Stops the loop so the Confirm/False Alarm buttons "
                                            "become clickable immediately.")
        lookup_services = st.checkbox("Query Overpass for nearest services", value=True,
                                      help="Untick to demo offline using the hardcoded fallbacks.")

    try:
        device = resolve_device(None if device_choice == "auto" else device_choice)
    except Exception as exc:  # torch missing
        st.sidebar.error(f"Could not resolve device: {exc}")
        return None

    st.sidebar.caption(f"Resolved device: `{device}`")
    if device == "cpu":
        st.sidebar.warning(DEVICE_NOTE)

    st.sidebar.header("Run")
    start = st.sidebar.button("Start", type="primary", use_container_width=True)
    col_stop, col_reset = st.sidebar.columns(2)
    stop = col_stop.button("Stop", use_container_width=True)
    reset = col_reset.button("Reset incident", use_container_width=True)

    if reset:
        reset_incident()
        st.rerun()
    if stop:
        # Only takes effect once the current loop has exited: Streamlit cannot
        # process a click while the script is still running.
        st.session_state.running = False

    return {
        "source": source,
        "label": label,
        "conf": conf,
        "gate_iou": gate_iou,
        "require_vehicle_overlap": require_vehicle_gate,
        "stride": stride,
        "resize_width": resize_width,
        "imgsz": imgsz,
        "display_fps": display_fps,
        "max_frames": int(max_frames),
        "window": window,
        "ratio": ratio,
        "cooldown": cooldown,
        "pause_on_confirm": pause_on_confirm,
        "lookup_services": lookup_services,
        "device": device,
        "half": device != "cpu",
        "start": start,
    }


def main() -> None:
    st.set_page_config(page_title="ResQVision — accident triage (prototype)",
                       page_icon="🚑", layout="wide")
    init_state()
    render_banner()
    st.title("ResQVision — accident detection & emergency triage (prototype)")
    st.caption("Frame-level detector + multi-frame gate. One incident at a time, "
               "human-confirmed, simulation only.")

    cfg = render_sidebar()

    col_video, col_panel = st.columns([3, 2])
    with col_video:
        st.subheader("Live video")
        video_slot = st.empty()
    with col_panel:
        panel_slot = st.empty()

    if cfg is None:
        with col_video:
            st.info("Configure a source in the sidebar.")
        with col_panel:
            render_panel()
        return

    weights = resolve_weights(None)
    model = None
    try:
        model = get_model(weights)
    except Exception as exc:  # missing deps, failed download, bad checkpoint
        st.session_state.model_error = f"{type(exc).__name__}: {exc}"
        st.error(
            "Could not load the model. Check that torch/ultralytics are installed "
            "(CUDA wheel first on Windows) and that the weights downloaded.\n\n"
            f"`{st.session_state.model_error}`"
        )

    if cfg["start"] and model is not None:
        st.session_state.running = True
        st.session_state.review = None
        st.session_state.notice = None

    if st.session_state.running and model is not None:
        st.session_state.incident = None
        with st.status("Running pipeline...", expanded=False) as run_status:
            run_stream(model, cfg["source"], cfg, video_slot, panel_slot)
            stats = st.session_state.last_stats or {}
            run_status.update(
                label=f"Stopped: {stats.get('stop_reason', 'n/a')} — "
                      f"{stats.get('processed', 0)} frames, "
                      f"{stats.get('hit_frames', 0)} hit frames, "
                      f"{stats.get('fps', 0.0):.1f} FPS",
                state="complete",
                expanded=True,
            )
        st.session_state.running = False
        st.rerun()

    with col_video:
        stats = st.session_state.last_stats
        caption = "annotated output (detections + track IDs)"
        if stats:
            caption += (f"  ·  {stats['processed']} frames  ·  {stats['hit_frames']} hit frames"
                        f"  ·  {stats['fps']:.1f} FPS  ·  {stats['stop_reason']}")
        render_video(st.session_state.last_frame, caption)
        if stats and stats["processed"] and not stats["confirmed"]:
            st.warning(
                "No incident was confirmed on this clip. That is the DESIRED result for "
                "jam_clip.mp4 — and a problem if you were running demo_clip.mp4 "
                "(try --gate-iou / the sidebar gate at 0.01)."
            )
        if stats and stats["confirmed"]:
            st.error("Incident confirmed. Review it in the panel and resolve it — "
                     "the confirmer ignores frames until you do.")

    with col_panel:
        # Clear whatever the run loop last wrote into the live placeholder, then
        # render the settled panel (and its buttons) into the same container.
        panel_slot.empty()
        with panel_slot.container():
            render_panel()


if __name__ == "__main__":
    main()
