#!/usr/bin/env python
"""
severity.py - ResQVision MVP, STEP 3b: rule-based severity estimate.

Two features, four lines of arithmetic, three labels. That is the whole module.

    estimate_severity(vehicle_count=4, confidence=0.92)  ->  "SEVERE"

SCORING (implemented exactly as specified):
    vehicle_count >= 3        -> +2
    vehicle_count == 2        -> +1
    confidence    >= 0.90     -> +2
    confidence    >= 0.75     -> +1
    score <= 1 -> MINOR | score <= 3 -> MODERATE | else -> SEVERE

WHAT THIS IS NOT:
    Not a medical assessment, not a legal assessment, not a crash-energy
    estimate. `vehicle_count` is "vehicles in frame", not "vehicles involved",
    and `confidence` is how sure a detector is about a *pattern it was trained
    on*, which is not the same as how bad the crash was. A 15-car traffic jam
    that trips the detector scores SEVERE. Keep the caption in the UI.

Run:
    python severity.py
"""

from __future__ import annotations

import argparse

# Single source of truth for the UI caption, so the disclaimer cannot drift
# between modules.
SEVERITY_NOTE = "Prototype heuristic — not a medically or legally validated assessment"

LABELS = ("MINOR", "MODERATE", "SEVERE")


def estimate_severity(vehicle_count: int, confidence: float) -> str:
    """Map (vehicles in frame, accident confidence) to MINOR/MODERATE/SEVERE."""
    score = 0
    if vehicle_count >= 3:
        score += 2
    elif vehicle_count == 2:
        score += 1
    if confidence >= 0.9:
        score += 2
    elif confidence >= 0.75:
        score += 1
    if score <= 1:
        return "MINOR"
    elif score <= 3:
        return "MODERATE"
    else:
        return "SEVERE"


def severity_score(vehicle_count: int, confidence: float) -> int:
    """The raw 0-4 score, exposed for display/debugging only."""
    score = 0
    if vehicle_count >= 3:
        score += 2
    elif vehicle_count == 2:
        score += 1
    if confidence >= 0.9:
        score += 2
    elif confidence >= 0.75:
        score += 1
    return score


def severity_breakdown(vehicle_count: int, confidence: float) -> dict:
    """Human-readable explanation of how the label was reached.

    Useful in the dashboard so a judge can see the rule instead of trusting a
    bare label.
    """
    vehicle_points = 2 if vehicle_count >= 3 else (1 if vehicle_count == 2 else 0)
    confidence_points = 2 if confidence >= 0.9 else (1 if confidence >= 0.75 else 0)
    return {
        "label": estimate_severity(vehicle_count, confidence),
        "score": vehicle_points + confidence_points,
        "vehicle_points": vehicle_points,
        "confidence_points": confidence_points,
        "vehicle_count": vehicle_count,
        "confidence": confidence,
        "note": SEVERITY_NOTE,
    }


# --------------------------------------------------------------------------- #
# Self-test: print the whole label surface, no video/deps required
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 3b: self-test + score table for the severity heuristic.",
    )
    parser.add_argument("--vehicle-counts", type=int, nargs="+", default=[1, 2, 3, 4, 6, 12, 20],
                        help="vehicle counts to tabulate")
    parser.add_argument("--confidences", type=float, nargs="+",
                        default=[0.0, 0.25, 0.50, 0.74, 0.75, 0.89, 0.90, 0.95, 0.99],
                        help="accident confidences to tabulate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    confidences = args.confidences

    print("=" * 78)
    print("Severity table   rows = vehicles in frame, cols = accident confidence")
    print("=" * 78)
    width = max(10, max(len(str(c)) for c in confidences) + 2)
    print(f"{'vehicles':>8} " + "".join(f"{c:>{width}.2f}" for c in confidences))
    print("-" * (9 + width * len(confidences)))
    for vc in args.vehicle_counts:
        row = f"{vc:>8} "
        for conf in confidences:
            row += f"{estimate_severity(vc, conf):>{width}}"
        print(row)

    print("\nScore contributions (vehicles | confidence -> total):")
    for vc in args.vehicle_counts:
        cells = []
        for conf in confidences:
            b = severity_breakdown(vc, conf)
            cells.append(f"{b['vehicle_points']}+{b['confidence_points']}={b['score']}")
        print(f"{vc:>8} " + "  ".join(f"{c:<8}" for c in cells))

    # Deterministic boundary assertions - the part worth failing loudly on.
    cases = [
        # (vehicle_count, confidence, expected)
        (1, 0.0, "MINOR"),      # score 0
        (1, 0.74, "MINOR"),     # score 0
        (1, 0.75, "MINOR"),     # score 1
        (2, 0.0, "MINOR"),      # score 1
        (2, 0.75, "MODERATE"),  # score 2
        (3, 0.0, "MODERATE"),   # score 2
        (1, 0.90, "MODERATE"),  # score 2
        (2, 0.90, "MODERATE"),  # score 3
        (3, 0.75, "MODERATE"),  # score 3
        (3, 0.90, "SEVERE"),    # score 4  <- the only SEVERE path
        (20, 0.91, "SEVERE"),   # score 4  <- also a traffic jam
    ]
    print("\nboundary assertions:")
    failures = []
    for vc, conf, expected in cases:
        got = estimate_severity(vc, conf)
        ok = got == expected
        print(f"  vehicles={vc:<3} conf={conf:<5} -> {got:<9} (expected {expected:<9}) "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"({vc}, {conf}) -> {got} != {expected}")

    # Monotonicity: raising either feature must never lower the score.
    monotonic = True
    for vc in range(0, 25):
        for conf in [i / 100 for i in range(0, 101, 5)]:
            s = severity_score(vc, conf)
            if severity_score(vc + 1, conf) < s:
                monotonic = False
            if severity_score(vc, min(1.0, conf + 0.05)) < s:
                monotonic = False
    print(f"\nmonotonicity over 0-24 vehicles x 0.00-1.00 confidence: "
          f"{'PASS' if monotonic else 'FAIL'}")
    if not monotonic:
        failures.append("monotonicity")

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILURES ({len(failures)}): " + "; ".join(failures))
        raise SystemExit(1)
    print(f"ALL {len(cases) + 1} CHECKS PASSED")
    print("=" * 78)
    print(f"\nRemember the caveat: {SEVERITY_NOTE}")


if __name__ == "__main__":
    main()
