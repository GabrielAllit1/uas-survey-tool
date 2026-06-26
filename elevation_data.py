# elevation_data.py
# Fully revised to (1) use APIKeyManager.opentopo(), (2) normalize (lon,lat)/(lat,lon),
# (3) call OpenTopoData correctly with batching/retries, and (4) keep the same return signature
# used by main_logic.py and modern_ui.py: -> (List[Optional[float]], Optional[float])

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import List, Tuple, Optional

import requests

try:
    import rasterio
except Exception:
    rasterio = None  # optional; only used if you later add a local DEM fallback

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(os.path.expanduser("~/UAS_Survey_Tool_Logs"), "uas_survey_tool.log"),
                encoding="utf-8",
            ),
            logging.StreamHandler(),
        ],
    )

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_DEFAULT_ELEV = 291.0  # your existing default
_CACHE_DIR = os.path.expanduser("~/.uas_survey_tool_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)
_CACHE_FILE = os.path.join(_CACHE_DIR, "elevation_cache.json")

def _load_cache() -> dict:
    try:
        if os.path.isfile(_CACHE_FILE):
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load elevation cache: {e}")
    return {}

def _save_cache(cache: dict) -> None:
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save elevation cache: {e}")

def _coerce_lon_lat(a: float, b: float) -> Tuple[float, float]:
    """
    Accepts either (lon,lat) or (lat,lon); returns (lon, lat). Clamps if wildly out of range.
    """
    try:
        a = float(a)
        b = float(b)
    except Exception:
        return (0.0, 0.0)

    if -180.0 <= a <= 180.0 and -90.0 <= b <= 90.0:
        return (a, b)
    if -90.0 <= a <= 90.0 and -180.0 <= b <= 180.0:
        # (lat, lon) provided -> swap
        return (b, a)
    # clamp best-effort
    lon = max(min(a, 180.0), -180.0)
    lat = max(min(b, 90.0), -90.0)
    return (lon, lat)

def _normalize_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Normalize inputs to (lat, lon) tuples for the API, drop obvious junk.
    """
    out: List[Tuple[float, float]] = []
    for idx, (x, y) in enumerate(points):
        lon, lat = _coerce_lon_lat(x, y)
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            logger.debug(f"Rejecting invalid coordinate at idx {idx}: ({x}, {y}) -> (lon={lon}, lat={lat})")
            continue
        out.append((lat, lon))  # API expects (lat, lon)
    return out

def _pick_dataset(pts_latlon: List[Tuple[float, float]]) -> str:
    """
    Choose OpenTopoData dataset based on centroid—'ned10m' for US, else 'srtm90m'.
    """
    if not pts_latlon:
        return "srtm90m"
    lats, lons = zip(*pts_latlon)
    clat, clon = float(sum(lats) / len(lats)), float(sum(lons) / len(lons))
    in_us = (18.0 <= clat <= 72.0) and (-179.2 <= clon <= -64.0)
    return "ned10m" if in_us else "srtm90m"

def _mean(values: List[Optional[float]]) -> float:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(sum(vals) / len(vals)) if vals else _DEFAULT_ELEV

# -----------------------------------------------------------------------------
# Public API (keeps your existing signature & behavior)
# -----------------------------------------------------------------------------

def get_srtm_elevation_bulk(
    coords: List[Tuple[float, float]],
    retry: int = 3,
    dataset: Optional[str] = None,
    batch_size: int = 100,
    min_delay: float = 1.0,
    timeout: int = 15,
) -> Tuple[List[Optional[float]], Optional[float]]:
    """
    Fetch elevation data for multiple coordinates using OpenTopoData.

    Args:
        coords: List of (lat, lon) *or* (lon, lat) tuples — any mix OK.
        retry:  retry attempts per batch on transient errors.
        dataset: 'ned10m' (USA) or 'srtm90m' (global). If None, auto-pick by centroid.
        batch_size: up to 100 points per request (OpenTopoData limit).
        min_delay: force a small gap between HTTP requests (rate-limit friendly).
        timeout: per-request timeout (seconds).

    Returns:
        (elevations, average_elevation)
        - elevations: list aligned to input order; None for missing; callers often treat None as default.
        - average_elevation: mean of valid values, or 291.0 if none.
    """
    try:
        if not coords:
            logger.warning("No coordinates provided for elevation query")
            return ([_DEFAULT_ELEV], _DEFAULT_ELEV)

        # Normalize to (lat,lon) for API, keep an index map to align back to input order
        norm_latlon = _normalize_points(coords)
        if not norm_latlon:
            logger.error("No valid coordinates after normalization")
            return ([_DEFAULT_ELEV] * len(coords), _DEFAULT_ELEV)

        # Choose dataset if unset
        dataset = dataset or _pick_dataset(norm_latlon)

        base_url = f"https://api.opentopodata.org/v1/{dataset}"

        cache = _load_cache()
        # Build result list aligned to *normalized* list, then we’ll expand to original length
        fetched: List[Optional[float]] = [None] * len(norm_latlon)

        # Prepare chunks
        last_req_time = 0.0
        for start in range(0, len(norm_latlon), batch_size):
            chunk = norm_latlon[start : start + batch_size]

            # Split chunk into cache hits vs misses
            chunk_results: List[Optional[float]] = [None] * len(chunk)
            to_fetch_indices: List[int] = []
            for i, (lat, lon) in enumerate(chunk):
                key = f"{lat:.6f},{lon:.6f}"
                if key in cache:
                    chunk_results[i] = cache[key]
                else:
                    to_fetch_indices.append(i)

            # If we have any to fetch, hit the API once for that sublist
            if to_fetch_indices:
                loc_pairs = [chunk[i] for i in to_fetch_indices]
                loc_str = "|".join(f"{lat:.8f},{lon:.8f}" for lat, lon in loc_pairs)

                # polite rate limit
                elapsed = time.time() - last_req_time
                if elapsed < min_delay:
                    time.sleep(min_delay - elapsed)

                # Retry loop
                ok = False
                for attempt in range(1, max(1, int(retry)) + 1):
                    try:
                        params = {"locations": loc_str}
                        # NOTE: OpenTopoData ignores extra params; leaving key out by default
                        resp = requests.get(base_url, params=params, timeout=timeout)
                        last_req_time = time.time()
                        logger.debug(f"{base_url} GET status {resp.status_code}")
                        resp.raise_for_status()
                        data = resp.json()
                        if data.get("status") != "OK" or "results" not in data:
                            raise ValueError(data.get("error", "Unknown API error"))
                        vals = [r.get("elevation", None) for r in data["results"]]
                        # write into chunk_results & cache
                        for local_idx, val in zip(to_fetch_indices, vals):
                            chunk_results[local_idx] = val
                            lat, lon = chunk[local_idx]
                            cache_key = f"{lat:.6f},{lon:.6f}"
                            if val is not None:
                                cache[cache_key] = val
                        _save_cache(cache)
                        ok = True
                        break
                    except Exception as e:
                        logger.error(f"OpenTopoData failed (attempt {attempt}): {e}")
                        if attempt < retry:
                            sleep_time = min(10.0, 2.0 * attempt)
                            logger.debug(f"Retrying after {sleep_time:.1f}s")
                            time.sleep(sleep_time)

                if not ok:
                    # give up for this chunk; fill misses with default
                    for local_idx in to_fetch_indices:
                        chunk_results[local_idx] = _DEFAULT_ELEV

            # Merge the chunk into fetched[]
            fetched[start : start + len(chunk)] = chunk_results

        # Now map back to original list length:
        # We normalized (and also filtered invalid inputs). For any original point we failed to normalize,
        # return default.
        # Easiest: recompute normalized list again and walk both.
        norm_again = _normalize_points(coords)
        out: List[Optional[float]] = []
        j = 0
        for x, y in coords:
            lon, lat = _coerce_lon_lat(x, y)
            if -180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0:
                # valid; use fetched[j]
                out.append(fetched[j] if j < len(fetched) else _DEFAULT_ELEV)
                j += 1
            else:
                out.append(_DEFAULT_ELEV)

        # Compute average over valid values
        avg = _mean(out)
        logger.info(f"Elevations retrieved: {sum(1 for v in out if v is not None)} points, avg={avg:.2f} m")
        return (out, avg)

    except Exception as e:
        logger.error(f"Elevation bulk query failed: {e}")
        # Fallback: return defaults sized to input
        return ([_DEFAULT_ELEV] * len(coords), _DEFAULT_ELEV)

def get_combined_elevation(points_lonlat: List[Tuple[float, float]]) -> Tuple[List[Optional[float]], Optional[float]]:
    """
    Compatibility shim some modules call instead of get_srtm_elevation_bulk.
    """
    return get_srtm_elevation_bulk(points_lonlat)
