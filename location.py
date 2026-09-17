#!/usr/bin/env python
"""
location.py - ResQVision MVP, STEP 4: demo location + nearest emergency service.

Two independent jobs, both intentionally tiny:

1. LOCATION. There is no GPS anywhere in this prototype. The incident location
   is a hardcoded demo camera position. Real deployments would map camera_id ->
   (lat, lon) in a config table; that table is out of scope here.

2. NEAREST SERVICE. One Overpass API query per amenity (hospital, police) with a
   5 km radius, called once per CONFIRMED INCIDENT - never per frame. Frames run
   at 5-10 Hz and Overpass is a free, shared, heavily rate-limited service; a
   query per frame would get you blocked within seconds.

If the query fails for any reason (offline, timeout, rate limit, malformed
response) the function returns a hardcoded fallback so the dashboard always has
something to render. The returned dict carries a "source" field so the UI can be
honest about whether the name came from OpenStreetMap or from the fallback.

    >>> get_nearest("hospital", 11.2588, 75.7804, FALLBACK_HOSPITAL)
    {'name': '...', 'lat': 11.2..., 'lon': 75.7..., 'source': 'overpass', ...}

Run:
    python location.py                 # live query for both amenities
    python location.py --verify        # also check elements[0] vs the true nearest
    python location.py --offline       # render the fallback path
"""

from __future__ import annotations

import argparse
import math
import time

import requests

# --------------------------------------------------------------------------- #
# Demo constants - single source of truth for the whole prototype
# --------------------------------------------------------------------------- #

DEMO_CAMERA = {"name": "Demo Camera 1", "lat": 11.2588, "lon": 75.7804}

FALLBACK_HOSPITAL = {"name": "City General Hospital", "lat": 11.2601, "lon": 75.7822}
FALLBACK_POLICE = {"name": "Central Police Station", "lat": 11.2570, "lon": 75.7790}

FALLBACKS = {
    "hospital": FALLBACK_HOSPITAL,
    "police": FALLBACK_POLICE,
}

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
SEARCH_RADIUS_M = 5000
REQUEST_TIMEOUT_S = 5

# Overpass' usage policy asks clients to identify themselves; requests' default
# UA is generic and more likely to be throttled.
USER_AGENT = "ResQVision-Hackathon-Prototype/0.1 (student project; contact: local)"


# --------------------------------------------------------------------------- #
# Nearest emergency service
# --------------------------------------------------------------------------- #

def get_nearest(amenity, lat, lon, fallback):
    """Nearest OSM feature with amenity={amenity} within 5 km, else the fallback.

    Args:
        amenity:  "hospital" or "police" (any OSM amenity value works).
        lat, lon: incident coordinates.
        fallback: dict returned when the lookup fails.

    Returns a dict with keys: name, lat, lon, source ("overpass" | "fallback"),
    distance_km, and error (only when source == "fallback").
    """
    result = None
    error = None

    try:
        query = (
            f'[out:json];node["amenity"="{amenity}"]'
            f'(around:{SEARCH_RADIUS_M},{lat},{lon});out body;'
        )
        resp = requests.post(
            OVERPASS_URL,
            data={"data": query},
            timeout=REQUEST_TIMEOUT_S,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        if elements:
            e = elements[0]
            result = {
                "name": e.get("tags", {}).get("name", amenity.title()),
                "lat": e["lat"],
                "lon": e["lon"],
            }
    except Exception as exc:
        # Offline, DNS failure, timeout, HTTP 429/504, non-JSON error page...
        # Any of these must degrade to the fallback rather than crash the run.
        error = f"{type(exc).__name__}: {exc}"

    if result is not None:
        return {
            "name": result["name"],
            "lat": result["lat"],
            "lon": result["lon"],
            "source": "overpass",
            "distance_km": round(haversine_km(lat, lon, result["lat"], result["lon"]), 3),
            "error": None,
        }

    return {
        "name": fallback["name"],
        "lat": fallback["lat"],
        "lon": fallback["lon"],
        "source": "fallback",
        "distance_km": round(haversine_km(lat, lon, fallback["lat"], fallback["lon"]), 3),
        "error": error or "no elements returned",
    }


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km. Distance is only shown to the user, never
    used to pick a service (see the note on elements[0] in `verify`)."""
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (math.sin(d_phi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2)
    return 2 * radius * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

def fetch_elements(lat, lon, amenity, query_body: str) -> tuple[list[dict], str | None]:
    """Run an arbitrary Overpass body query; return (elements, error)."""
    try:
        resp = requests.post(
            OVERPASS_URL,
            data={"data": query_body},
            timeout=REQUEST_TIMEOUT_S,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
        return resp.json().get("elements", []), None
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"


def verify(lat: float, lon: float) -> int:
    """Check two assumptions baked into get_nearest().

    Assumption A: elements[0] is the nearest. Overpass makes NO ordering
    guarantee - response order is effectively arbitrary (internal node id order),
    so elements[0] is whatever came back first, not the closest.

    Assumption B: `node[...]` finds the hospitals/police stations. It only finds
    ones mapped as point nodes; anything mapped as a building outline (a `way`)
    or a multipolygon (a `relation`) is invisible to that query.
    """
    print("=" * 78)
    print("ASSUMPTION CHECK - is elements[0] the nearest, and does node[] find them?")
    print("=" * 78)

    reports = []
    for amenity in ("hospital", "police"):
        node_query = (
            f'[out:json];node["amenity"="{amenity}"]'
            f'(around:{SEARCH_RADIUS_M},{lat},{lon});out body;'
        )
        nwr_query = (
            f'[out:json];nwr["amenity"="{amenity}"]'
            f'(around:{SEARCH_RADIUS_M},{lat},{lon});out center;'
        )
        nodes, node_err = fetch_elements(lat, lon, amenity, node_query)
        nwr, nwr_err = fetch_elements(lat, lon, amenity, nwr_query)

        def coords_of(e):
            return (e["lat"], e["lon"]) if "lat" in e and "lon" in e else (
                e.get("center", {}).get("lat"), e.get("center", {}).get("lon"))

        rows = []
        for label, elements, err in (("node[]", nodes, node_err), ("nwr[]", nwr, nwr_err)):
            if err:
                rows.append((label, f"ERROR {err}", None, None))
                continue
            located = [(e, *coords_of(e)) for e in elements]
            located = [(e, la, lo) for e, la, lo in located if la is not None and lo is not None]
            distances = sorted(haversine_km(lat, lon, la, lo) for _, la, lo in located)
            first_name = located[0][0].get("tags", {}).get("name", "(unnamed)") if located else None
            first_dist = distances[0] if distances else None
            rows.append((label, f"{len(located)} element(s)", first_name, first_dist))
            if label == "nwr[]" and located and first_dist is not None:
                true_nearest = min(located, key=lambda t: haversine_km(lat, lon, t[1], t[2]))
                true_name = true_nearest[0].get("tags", {}).get("name", "(unnamed)")
                true_dist = haversine_km(lat, lon, true_nearest[1], true_nearest[2])
                reports.append((amenity, located, true_name, true_dist))

        print(f"\n{amenity!r}:")
        for label, count, first_name, first_dist in rows:
            if first_dist is None:
                print(f"   {label:<8} {count}")
            else:
                print(f"   {label:<8} {count:<14} first element: {first_name!r} "
                      f"at {first_dist:.2f} km")
        for amenity_name, located, true_name, true_dist in reports:
            if amenity_name != amenity:
                continue
            first = located[0]
            first_name = first[0].get("tags", {}).get("name", "(unnamed)")
            first_dist = haversine_km(lat, lon, first[1], first[2])
            verdict = "same" if abs(first_dist - true_dist) < 1e-9 else "DIFFERENT"
            print(f"   -> nearest by distance: {true_name!r} at {true_dist:.2f} km; "
                  f"elements[0] is {verdict}")
            if verdict == "DIFFERENT":
                print(f"      get_nearest() returns {first_name!r} ({first_dist:.2f} km) "
                      f"instead of the closest one.")

    print("\n" + "-" * 78)
    print("Interpretation:")
    print("  * 'DIFFERENT' above means the returned hospital/police station is a real")
    print("    OSM feature but not necessarily the closest one. The distance shown in")
    print("    the UI is still correct for the name shown, so the demo stays honest.")
    print("  * If nwr[] finds more elements than node[], some nearby services are")
    print("    mapped as ways/relations and are invisible to the specified query.")
    print("=" * 78)
    return 0


# --------------------------------------------------------------------------- #
# Standalone runner
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4: standalone test of the demo location + nearest-service lookup.",
    )
    parser.add_argument("--offline", action="store_true",
                        help="skip the network entirely and show the fallback path")
    parser.add_argument("--verify", action="store_true",
                        help="extra diagnostics on Overpass result ordering and node-vs-way coverage")
    parser.add_argument("--lat", type=float, default=DEMO_CAMERA["lat"])
    parser.add_argument("--lon", type=float, default=DEMO_CAMERA["lon"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lat, lon = args.lat, args.lon

    print("=" * 78)
    print("STEP 4 - location lookup standalone test")
    print("=" * 78)
    print(f"demo camera : {DEMO_CAMERA['name']}  ({DEMO_CAMERA['lat']}, {DEMO_CAMERA['lon']})")
    print(f"query point : ({lat}, {lon})")
    print(f"radius      : {SEARCH_RADIUS_M} m   endpoint: {OVERPASS_URL}")
    print(f"timeout     : {REQUEST_TIMEOUT_S} s per query")
    if args.offline:
        print("mode        : OFFLINE (network calls disabled)")
    print()

    if args.verify and not args.offline:
        verify(lat, lon)
        print()

    for amenity in ("hospital", "police"):
        fallback = FALLBACKS[amenity]
        if args.offline:
            result = {
                "name": fallback["name"], "lat": fallback["lat"], "lon": fallback["lon"],
                "source": "fallback",
                "distance_km": round(haversine_km(lat, lon, fallback["lat"], fallback["lon"]), 3),
                "error": "offline mode",
            }
            elapsed = 0.0
        else:
            tick = time.perf_counter()
            result = get_nearest(amenity, lat, lon, fallback)
            elapsed = time.perf_counter() - tick

        print(f"{amenity.upper():<9} source={result['source']:<9} "
              f"in {elapsed * 1000:6.1f} ms")
        print(f"          name      : {result['name']}")
        print(f"          coords    : ({result['lat']}, {result['lon']})")
        print(f"          distance  : {result['distance_km']} km from the camera")
        if result["source"] == "fallback":
            print(f"          -> FALLBACK USED. reason: {result['error']}")
            print(f"          -> the dashboard will show this hardcoded name; the UI")
            print(f"             displays source={result['source']!r} so it is not misleading.")
        else:
            print(f"          -> live result from OpenStreetMap.")
        print()

    print("-" * 78)
    print("Reminders:")
    print("  * Called once per CONFIRMED incident, never per frame.")
    print("  * Overpass is free and shared. 5 s timeout can and will expire under load;")
    print("    fallbacks exist so a 504 during the demo does not blank the dashboard.")
    print("  * Coordinates are OpenStreetMap data (ODbL) - attribute it if you show the map.")
    print("=" * 78)


if __name__ == "__main__":
    main()
