# ResQVision — Accident Detection & Emergency Triage (Prototype)

> ⚠️ **SIMULATION ONLY.** This is a hackathon MVP. It does not contact any real
> hospital, police department, or dispatch system. There is no database —
> incidents live in memory for a single browser tab and vanish when it closes.
> The "Confirm Alert" button prints a simulated-notification message and calls
> no external API.

ResQVision watches a traffic video feed, uses a YOLO11x object detector to spot
accident-like frames, requires the signal to be *sustained* (not a one-frame
fluke) before it calls anything an "incident," estimates a rough severity, looks
up the nearest hospital/police station on OpenStreetMap, and hands the whole
thing to a human for confirmation before anything is "sent." It ships as a
Streamlit dashboard, but every stage also runs standalone from the command line.

## How it works

```
video / webcam
      │
      ▼
detector.py      → YOLO11x inference per frame, accident<->vehicle IoU gate
      │             (FrameStream, Detection, load_model, draw_detections)
      ▼
tracker.py       → wraps the detector with ByteTrack so vehicles keep a
      │             stable ID across frames (VehicleTracker.analyze → FrameAnalysis)
      ▼
confirm.py       → rolling-window "sustained signal" gate — needs hit_ratio
      │             of the last window_size frames to actually confirm
      ▼
severity.py      → rule-based MINOR / MODERATE / SEVERE label from
      │             vehicle count + detector confidence
      ▼
location.py      → nearest hospital & police station via the OpenStreetMap
      │             Overpass API, once per incident, with hardcoded fallbacks
      ▼
app.py           → Streamlit dashboard: live video, incident panel, map,
                    and the human Confirm / False-Alarm buttons
```

Nothing in this pipeline is a medical, legal, or dispatch-grade judgment — see
[Scope & limitations](#scope--limitations) below.

## Features

- **Detection** — YOLO11x fine-tuned on an `accident` / `vehicle` two-class
  problem ([`Enos-123/traffic-accident-detection-yolo11x`](https://huggingface.co/Enos-123/traffic-accident-detection-yolo11x)
  on Hugging Face), downloaded automatically on first run.
- **Tracking** — ByteTrack (via Ultralytics) gives each vehicle a stable ID so
  the same car isn't double-counted frame to frame.
- **Confirmation gate** — a single frame saying "accident" proves nothing; a
  parked wreck, a car with its hazards on, or one noisy frame will all trip
  the raw detector. `confirm.py` only raises an incident once a configurable
  fraction of a rolling window of frames agree, then enforces a cooldown so
  one crash can't spam multiple incidents.
- **Severity heuristic** — a transparent, four-point rule
  (`vehicle_count` + `confidence` → score → label) — explicitly *not* a
  medical or crash-severity assessment.
- **Nearest services** — queries OpenStreetMap's Overpass API for the closest
  hospital and police station to a fixed demo camera location, with
  hardcoded fallbacks if the lookup fails or is disabled.
- **Human-in-the-loop** — every incident requires a person to click
  **Confirm Alert** or **False Alarm** before it's treated as resolved;
  nothing is dispatched automatically.
- **Two demo clips are used to validate the pipeline on purpose**:
  `demo_clip.mp4` (should confirm) and `jam_clip.mp4`, a dense traffic jam
  that looks alarming but should *not* confirm.

## Project structure

| File | Role |
|---|---|
| `app.py` | Streamlit dashboard — ties every module together into a UI |
| `detector.py` | Model loading, video I/O, detection → `Detection` objects, IoU gate, drawing/HUD, standalone CLI runner |
| `tracker.py` | Wraps the detector with object tracking; exposes `VehicleTracker.analyze(frame)` |
| `confirm.py` | `SimpleConfirmer` — the rolling-window, cooldown-gated confirmation logic, with a built-in self-test |
| `severity.py` | `estimate_severity()` / `severity_breakdown()` — the MINOR/MODERATE/SEVERE heuristic, with a built-in self-test |
| `location.py` | Overpass API lookup for the nearest hospital/police station, with hardcoded fallbacks |
| `test_model.py` | Sanity-checks the detection model in isolation |
| `requirements.txt` | Python dependencies (see the CUDA note inside it) |
| `.gitignore` | Standard Python/venv/data ignores |

> Note: `data/` (video clips, downloaded weights) is not tracked in the repo —
> see [Setup](#setup) for where to put your own clips.

## Setup

**Requirements:** Python 3.13 (verified), pip.

```bash
git clone https://github.com/pssanjay006/hack-the-detector.git
cd hack-the-detector
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

**Install PyTorch first if you have a GPU.** Plain `pip install torch` gives
you the CPU-only wheel on Windows, which silently runs everything in FP32
instead of FP16 on your CUDA device:

```bash
# GPU (CUDA 12.4, Windows/Linux)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU only
pip install torch torchvision
```

Then the rest of the dependencies:

```bash
pip install -r requirements.txt
```

Verify your GPU is actually being used:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

**Add video clips.** Create a `data/` folder in the repo root and drop in your
own clips, or provide `data/demo_clip.mp4` / `data/jam_clip.mp4` if you have
the project's original test clips — the dashboard auto-detects them, and you
can also upload a clip or use a webcam directly from the UI. The YOLO
weights (~114 MB) download automatically from Hugging Face the first time you
run anything that needs the model.

## Running it

**Dashboard (recommended):**

```bash
streamlit run app.py
```

Pick a source in the sidebar, hit **Start**, and watch the incident panel.
Sidebar controls let you tune detector confidence, the accident↔vehicle IoU
gate, frame stride/resolution, and the confirmation window/ratio/cooldown —
each with an explanation of what it does and why the default is set that way.

**Detector only** (useful for eyeballing detection quality before touching
tracking or the UI — writes an annotated video to `data/`):

```bash
python detector.py --source data/demo_clip.mp4
python detector.py --source data/jam_clip.mp4 --conf 0.25
python detector.py --source 0 --max-frames 200        # webcam
```

**Confirmation logic self-test** (no video or GPU required):

```bash
python confirm.py
python confirm.py --window-size 8 --hit-ratio 0.75 --cooldown 30
```

**Severity heuristic self-test / score table** (no dependencies beyond stdlib):

```bash
python severity.py
```

## Tuning notes worth knowing

- The **accident↔vehicle IoU gate** defaults conservatively, but measured
  overlap between a real accident box and a real vehicle box on the demo
  clips is usually only ~0.02–0.05 IoU. A gate of 0.3 can block almost every
  frame; the dashboard's sidebar documents this and defaults sensibly.
- Whether "require accident↔vehicle overlap" is ticked changes results
  dramatically: with it on, the model's `vehicle` class needs to actually
  fire near the accident box on the same frame, which some clips (and some
  camera angles) rarely produce. Toggle it off to rely on the accident box
  alone.
- `confirm.py` measures **persistence, not onset**. A parked wreck or a
  stalled car with its hazards on will look identical to a real, ongoing
  crash to this module — that's why a human confirmation step exists.

## Scope & limitations

This is a hackathon MVP with a deliberately narrow scope:

- One active incident at a time.
- One hardcoded demo camera location (no multi-camera support).
- No evidence clips, no face or license-plate handling.
- No real dispatch, notification, or database — everything lives in the
  browser tab's session state.
- Severity is a simple, transparent heuristic — **not** a medical, legal, or
  insurance-grade assessment.

## Model credit

Detection weights: [`Enos-123/traffic-accident-detection-yolo11x`](https://huggingface.co/Enos-123/traffic-accident-detection-yolo11x)
on Hugging Face (YOLO11x fine-tuned for `accident` / `vehicle` classes).
