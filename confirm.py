#!/usr/bin/env python
"""
confirm.py - ResQVision MVP, STEP 3a: multi-frame accident confirmation.

This is the module that decides "a single frame said accident" is not the same
as "an accident happened". It is a rolling-window hit counter: it keeps the last
`window_size` frame outcomes and confirms once at least `hit_ratio` of them were
accident hits.

It is deliberately dumb and deliberately stateless-with-respect-to-video: it
takes booleans in and returns a boolean out, so it can be tested exhaustively
with fake sequences and zero video (see the `__main__` block).

    confirmer = SimpleConfirmer(window_size=10, hit_ratio=0.8, cooldown_sec=30)
    for frame in stream:
        analysis = tracker.analyze(frame)
        if confirmer.update(analysis.accident_detected):
            ...  # exactly once per incident

IMPORTANT SEMANTICS (read before tuning):
    This confirmer measures PERSISTENCE, not ONSET. A wrecked car parked on a
    shoulder, a car broken down with hazards on, or a vehicle the detector
    simply likes will produce 10/10 hits and therefore CONFIRM, as will a real
    crash. The confirmer cannot distinguish those cases, because per-frame
    object detection carries no notion of "this crash just started". That is
    what the human confirmation button and the jam_clip.mp4 negative test are
    for. Do not describe this module as "accident verification" on stage; call
    it what it is, a sustained-signal gate.

Run:
    python confirm.py
    python confirm.py --window-size 8 --hit-ratio 0.75 --cooldown 30
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque


class SimpleConfirmer:
    """Rolling-window accident confirmer with a post-confirmation cooldown.

    Args:
        window_size: how many frame outcomes the window holds.
        hit_ratio:   fraction of the window that must be hits to confirm.
        cooldown_sec: seconds during which `update()` refuses to even record
                      frames after a confirmation, so one crash cannot produce
                      a burst of incidents.
    """

    def __init__(self, window_size: int = 10, hit_ratio: float = 0.8, cooldown_sec: float = 30):
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not 0.0 < hit_ratio <= 1.0:
            raise ValueError("hit_ratio must be in (0, 1]")
        self.window = deque(maxlen=window_size)
        self.hit_ratio = hit_ratio
        self.cooldown_sec = cooldown_sec
        self.cooldown_until = 0.0
        self.confirmed = False
        # Introspection for the dashboard / test output. Not used in the logic.
        self.frames_seen = 0
        self.confirmations = 0
        self.last_confirmed_at: float | None = None

    def update(self, accident_detected_this_frame: bool) -> bool:
        """Record one frame. Return True exactly once per confirmed incident.

        While the cooldown is active the frame is DROPPED (not recorded as a
        miss), so a cooldown period does not dilute the window with zeros.
        """
        now = time.time()
        if now < self.cooldown_until:
            return False
        self.window.append(1 if accident_detected_this_frame else 0)
        self.frames_seen += 1
        if len(self.window) == self.window.maxlen:
            ratio = sum(self.window) / len(self.window)
            if ratio >= self.hit_ratio and not self.confirmed:
                self.confirmed = True
                self.cooldown_until = now + self.cooldown_sec
                self.confirmations += 1
                self.last_confirmed_at = now
                return True
        return False

    def update_frame(self, analysis) -> bool:
        """Convenience adapter: feed a tracker.FrameAnalysis directly.

        Works by attribute access only, so this module never imports torch.
        """
        return self.update(bool(analysis.accident_detected))

    def reset(self):
        """Clear window and latch. Call after the human resolves the incident."""
        self.window.clear()
        self.confirmed = False
        self.cooldown_until = 0.0

    def status(self) -> dict:
        """Read-only snapshot for the dashboard. Does not mutate anything."""
        window_frames = len(self.window)
        hits = sum(self.window)
        now = time.time()
        return {
            "confirmed": self.confirmed,
            "window_size": self.window.maxlen,
            "window_frames": window_frames,
            "hits": hits,
            "ratio": (hits / window_frames) if window_frames else 0.0,
            "hit_ratio": self.hit_ratio,
            "hit_sequence": "".join(str(v) for v in self.window),
            "in_cooldown": now < self.cooldown_until,
            "cooldown_remaining": max(0.0, self.cooldown_until - now),
            "confirmations": self.confirmations,
        }


# --------------------------------------------------------------------------- #
# Self-test: no video, no torch, no numpy - just fake hit sequences
# --------------------------------------------------------------------------- #

def run_scenario(
    name: str,
    hits: list[bool],
    expect_confirms: int,
    window_size: int,
    hit_ratio: float,
    cooldown_sec: float,
    expect_confirmed_at: int | None = None,
) -> dict:
    """Feed a fake hit sequence through a fresh confirmer and report."""
    confirmer = SimpleConfirmer(
        window_size=window_size, hit_ratio=hit_ratio, cooldown_sec=cooldown_sec
    )
    confirmed_at: int | None = None
    confirms = 0
    for index, hit in enumerate(hits, start=1):
        if confirmer.update(hit):
            confirms += 1
            if confirmed_at is None:
                confirmed_at = index

    passed = confirms == expect_confirms
    if expect_confirmed_at is not None:
        passed = passed and confirmed_at == expect_confirmed_at

    return {
        "name": name,
        "frames": len(hits),
        "confirms": confirms,
        "confirmed_at": confirmed_at,
        "expected": expect_confirms,
        "expected_at": expect_confirmed_at,
        "passed": passed,
        "final": confirmer.status(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 3a: self-test for the rolling-window accident confirmer.",
    )
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--hit-ratio", type=float, default=0.8)
    parser.add_argument("--cooldown", type=float, default=30.0,
                        help="cooldown seconds; use 0 to disable it in the tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    w, r = args.window_size, args.hit_ratio
    # A real cooldown would make later scenarios in the same test unrunnable, so
    # the functional scenarios use 0 and the cooldown is asserted separately.
    c = 0.0

    # Hits required inside a full window. With the defaults (w=10, r=0.8) this
    # is 8. Every scenario below is built from it rather than hardcoded, so the
    # assertions stay correct if you pass --window-size/--hit-ratio.
    needed = math.ceil(r * w)
    below = max(0, needed - 1)

    scenarios = [
        # --- should CONFIRM -------------------------------------------------
        run_scenario(f"exactly {needed} hits inside the window, then quiet",
                     [True] * needed + [False] * (w - needed) + [False] * 3,
                     1, w, r, c, expect_confirmed_at=w),
        run_scenario("steady accident (all hits)", [True] * 40, 1, w, r, c,
                     expect_confirmed_at=w),
        run_scenario("ramp: quiet then sustained hits (spec example shape)",
                     [False] * (w - needed) + [True] * needed + [True] * 5,
                     1, w, r, c, expect_confirmed_at=w),
        run_scenario("exactly window_size hits", [True] * w, 1, w, r, c,
                     expect_confirmed_at=w),
        # --- should NOT confirm --------------------------------------------
        run_scenario("single isolated false-positive frame",
                     [False] * 5 + [True] + [False] * 24, 0, w, r, c),
        run_scenario("jam-like flicker (~50% hits)", [True, False] * 20, 0, w, r, c),
        run_scenario(f"periodic {below}/{w} hits, forever",
                     ([True] * below + [False] * (w - below)) * 4, 0, w, r, c),
        run_scenario("two separated short bursts", ([True] * 5 + [False] * 10) * 4,
                     0, w, r, c),
        run_scenario("no detections at all", [False] * 40, 0, w, r, c),
    ]

    print("=" * 78)
    print(f"SimpleConfirmer self-test   window_size={w}  hit_ratio={r}  cooldown=0")
    print(f"confirmation requires {needed}/{w} hits in the window")
    print("=" * 78)
    header = f"{'scenario':<52} {'frames':>6} {'conf':>5} {'at':>4} {'exp':>4} {'result':>7}"
    print(header)
    print("-" * len(header))
    failures = []
    for s in scenarios:
        at = "-" if s["confirmed_at"] is None else str(s["confirmed_at"])
        print(f"{s['name']:<52} {s['frames']:>6} {s['confirms']:>5} {at:>4} "
              f"{s['expected']:>4} {'PASS' if s['passed'] else 'FAIL':>7}")
        if not s["passed"]:
            failures.append(s["name"])

    # Cooldown: one confirmation, then a burst of further hits must stay silent.
    print("\ncooldown behaviour (cooldown_sec=30):")
    confirmer = SimpleConfirmer(window_size=w, hit_ratio=r, cooldown_sec=30)
    first_at = None
    confirms = 0
    for i in range(1, 201):
        if confirmer.update(True):
            confirms += 1
            first_at = first_at or i
    st = confirmer.status()
    cooldown_ok = confirms == 1 and st["in_cooldown"]
    print(f"  200 consecutive hit frames -> {confirms} confirmation(s) at frame {first_at}, "
          f"cooldown active={st['in_cooldown']} "
          f"({st['cooldown_remaining']:.1f}s left) -> {'PASS' if cooldown_ok else 'FAIL'}")
    if not cooldown_ok:
        failures.append("cooldown suppresses repeats")

    # reset() must re-arm the confirmer for the next incident.
    print("\nreset behaviour:")
    confirmer = SimpleConfirmer(window_size=w, hit_ratio=r, cooldown_sec=0)
    before = sum(confirmer.update(True) for _ in range(w))
    window_cleared = len(confirmer.window)  # after the loop, window is full
    confirmer.reset()
    cleared = len(confirmer.window) == 0 and not confirmer.confirmed
    after = sum(confirmer.update(True) for _ in range(w))
    reset_ok = before == 1 and after == 1 and cleared
    print(f"  confirm -> reset -> confirm again: before={before} after={after}, "
          f"window emptied by reset={cleared} (window was {window_cleared} deep) "
          f"-> {'PASS' if reset_ok else 'FAIL'}")
    if not reset_ok:
        failures.append("reset re-arms")

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILURES ({len(failures)}): " + "; ".join(failures))
        raise SystemExit(1)
    print(f"ALL {len(scenarios) + 2} CHECKS PASSED")
    print("=" * 78)
    print(
        "\nReminder: all of the above are PASSING while the module still cannot\n"
        "tell a fresh crash from a car that crashed yesterday. Persistence only."
    )


if __name__ == "__main__":
    main()
