import requests
import time
import numpy as np

def detect_elevation_cycles(elevation_data):
    """Detect cyclic patterns in elevation data (example: using mod-9 fibs)."""
    fib_mod9 = [1, 1, 2, 3, 5, 8, 4, 3, 7, 1, 8, 0, 8, 8, 7, 6, 4, 1, 5, 6, 2, 8, 1, 0]
    max_elev = max(elevation_data) if elevation_data else 1
    cycle_score = sum(fib_mod9[i % 24] * elev / max_elev for i, elev in enumerate(elevation_data)) / (len(elevation_data) or 1)
    return cycle_score

def _validate_and_format_latlon(latlon_points):
    """Ensure all are (lat, lon) with valid ranges, auto-correct if swapped."""
    formatted = []
    for idx, pt in enumerate(latlon_points):
        if len(pt) != 2:
            continue
        lat, lon = pt
        # If out of bounds, try to swap
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            lat, lon = lon, lat
        # Now validate
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            formatted.append((lat, lon))
    return formatted

def get_elevation_data(latlon_points, retry=3, delay=2):
    """
    Query OpenTopoData/SRTM API for a list of (lat, lon) points.
    Returns list of elevations (in meters, float).
    """
    api_url = "https://api.opentopodata.org/v1/srtm90m"
    coords = _validate_and_format_latlon(latlon_points)
    elevations = []
    attempt = 0

    while attempt < retry:
        try:
            # Batch in chunks of 100 for OpenTopoData API
            for i in range(0, len(coords), 100):
                chunk = coords[i:i + 100]
                payload = {"locations": [{"latitude": lat, "longitude": lon} for lat, lon in chunk]}
                resp = requests.post(api_url, json=payload, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    for result in data.get("results", []):
                        elevations.append(result.get("elevation", 0.0))
                else:
                    print(f"[SRTM API] Bad response: {resp.status_code}")
                    elevations.extend([0.0] * len(chunk))
                time.sleep(delay)
            print(f"[DEBUG] Fetched {len(elevations)} elevations")
            return elevations
        except Exception as e:
            print(f"[SRTM API] Failed: {e}")
            attempt += 1
            if attempt < retry:
                print(f"[INFO] Retrying... Attempt {attempt + 1}/{retry}")
                time.sleep(delay * attempt)
    return [0.0] * len(latlon_points)
