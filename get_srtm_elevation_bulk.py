import os
import time
import logging
import requests
import numpy as np
import json
from typing import List, Tuple, Union, Optional

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

US_BOUNDS = {
    "lat_min": 18.0,
    "lat_max": 72.0,
    "lon_min": -179.2,
    "lon_max": -64.0
}

DATASETS = {
    'us': 'ned10m',
    'global': 'srtm90m'
}

CACHE_DIR = os.path.expanduser("~/.uas_survey_tool_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "elevation_cache.json")

def load_cache() -> dict:
    """Load elevation cache from JSON file."""
    try:
        if os.path.isfile(CACHE_FILE):
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        logger.debug(f"No cache file found at {CACHE_FILE}")
    except Exception as e:
        logger.warning(f"Failed to load elevation cache: {e}")
    return {}

def save_cache(cache: dict) -> None:
    """Save elevation cache to JSON file."""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
        logger.debug(f"Saved elevation cache to {CACHE_FILE}")
    except Exception as e:
        logger.warning(f"Failed to save elevation cache: {e}")

def is_point_in_us(lat: float, lon: float) -> bool:
    """Check if a point is within US bounds."""
    return (
        US_BOUNDS['lat_min'] <= lat <= US_BOUNDS['lat_max']
        and US_BOUNDS['lon_min'] <= lon <= US_BOUNDS['lon_max']
    )

def pick_elevation_dataset(pts: List[Tuple[float, float]]) -> str:
    """Select appropriate dataset based on centroid location."""
    if not pts:
        logger.warning("No points provided to pick dataset, defaulting to SRTM90m.")
        return DATASETS['global']
    lats, lons = zip(*pts)
    centroid_lat, centroid_lon = float(np.mean(lats)), float(np.mean(lons))
    if is_point_in_us(centroid_lat, centroid_lon):
        logger.info("Centroid detected in US bounds. Using NED10m.")
        return DATASETS['us']
    else:
        logger.info("Centroid outside US. Using SRTM90m.")
        return DATASETS['global']

def validate_latlon(latlon_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Validate and correct lat/lon points, handling potential swaps."""
    valid_points = []
    for idx, (x, y) in enumerate(latlon_points):
        try:
            x, y = float(x), float(y)
            # Check if (x, y) is (lat, lon)
            if -90 <= x <= 90 and -180 <= y <= 180:
                lat, lon = x, y
            # Check if (x, y) is (lon, lat)
            elif -90 <= y <= 90 and -180 <= x <= 180:
                lat, lon = y, x
                logger.debug(f"Swapped coordinates for point #{idx}: ({x}, {y}) -> (lat={lat}, lon={lon})")
            else:
                logger.error(f"Invalid coordinate pair #{idx}: ({x}, {y}) - outside valid bounds")
                continue
            valid_points.append((lat, lon))
            logger.debug(f"Validated point #{idx}: (lat={lat:.6f}, lon={lon:.6f})")
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid coordinate format for point #{idx}: ({x}, {y}) - {e}")
            continue
    if not valid_points:
        logger.error("No valid coordinates after validation")
    return valid_points

def get_srtm_elevation_bulk(
    points: List[Tuple[float, float]],
    dataset: str = None,
    batch_size: int = 100,
    max_retries: int = 3,
    method: str = 'GET',
    sleep_base: float = 2.0,
    max_sleep: float = 10.0,
    min_delay: float = 1.0
) -> Tuple[List[Optional[float]], float]:
    """
    Query elevation data from OpenTopoData API with batching and caching.

    :param points: List of (lat, lon) tuples.
    :param dataset: Dataset to use ('ned10m' or 'srtm90m'). If None, auto-select based on centroid.
    :param batch_size: Number of points per API request (max 100 per OpenTopoData).
    :param max_retries: Number of retry attempts for failed requests.
    :param method: HTTP method ('GET' or 'POST').
    :param sleep_base: Base sleep time for exponential backoff.
    :param max_sleep: Maximum sleep time for retries.
    :param min_delay: Minimum delay between requests to respect API rate limit (1 call/sec).
    :return: Tuple of (elevation list, average elevation).
    """
    logger.debug(f"Processing {len(points)} points for elevation query")
    
    # Validate and format points
    valid_points = validate_latlon(points)
    if not valid_points:
        logger.error("No valid points provided for elevation query")
        return [291.0] * len(points), 291.0

    # Load cache
    cache = load_cache()

    # Check cache for existing elevations
    to_query = []
    cached_elevations = [None] * len(valid_points)
    for i, (lat, lon) in enumerate(valid_points):
        cache_key = f"{lat:.6f},{lon:.6f}"
        if cache_key in cache:
            cached_elevations[i] = cache[cache_key]
            logger.debug(f"Cache hit for {cache_key}: {cached_elevations[i]} m")
        else:
            to_query.append((lat, lon))

    # Select dataset if not specified
    if dataset is None:
        dataset = pick_elevation_dataset(valid_points)
    
    # Query OpenTopoData for remaining points
    results = []
    url = f"https://api.opentopodata.org/v1/{dataset}"
    last_request_time = 0

    for i in range(0, len(to_query), batch_size):
        chunk = to_query[i:i + batch_size]
        loc_str = "|".join(f"{lat},{lon}" for lat, lon in chunk)
        
        for attempt in range(1, max_retries + 1):
            try:
                # Enforce minimum delay to respect API rate limit
                elapsed = time.time() - last_request_time
                if elapsed < min_delay:
                    time.sleep(min_delay - elapsed)
                
                logger.info(f"Trying {dataset} API for {len(chunk)} points, attempt {attempt}")
                logger.debug(f"Starting new HTTPS connection to {url}")
                resp = requests.get(url, params={"locations": loc_str}, timeout=15)
                last_request_time = time.time()
                
                logger.debug(f"{url} \"GET /v1/{dataset}?locations={loc_str} HTTP/1.1\" {resp.status_code} None")
                resp.raise_for_status()
                
                data = resp.json()
                if data.get("status") != "OK":
                    raise ValueError(data.get("error", "Unknown API error"))
                
                elevs = [r.get("elevation", None) for r in data["results"]]
                for (lat, lon), elev in zip(chunk, elevs):
                    if elev is not None:
                        cache_key = f"{lat:.6f},{lon:.6f}"
                        cache[cache_key] = elev
                        logger.debug(f"DEM elevation from API at (lat={lat:.6f}, lon={lon:.6f}): {elev:.2f} m")
                
                results.extend(elevs)
                logger.debug(f"Retrieved {len(elevs)} elevations for batch {i//batch_size + 1}")
                logger.info(f"Elevations retrieved: {len([e for e in elevs if e is not None])} valid points, avg={np.mean([e for e in elevs if e is not None]):.2f} m")
                
                save_cache(cache)
                break
            except Exception as e:
                logger.error(f"OpenTopoData failed (attempt {attempt}): {e}")
                if attempt < max_retries:
                    sleep_time = min(max_sleep, sleep_base * (2 ** (attempt - 1)))
                    logger.debug(f"Retrying after {sleep_time:.2f} seconds")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Batch {i//batch_size + 1} exhausted retries. Using default elevation 291.0 m")
                    results.extend([291.0] * len(chunk))
    
    # Merge cached and fetched elevations
    merged = []
    fetch_idx = 0
    for v in cached_elevations:
        if v is not None:
            merged.append(v)
        else:
            merged.append(results[fetch_idx] if fetch_idx < len(results) else 291.0)
            fetch_idx += 1
    
    # Calculate average elevation for valid points
    valid_elevs = [e for e in merged if e is not None and e > 0]
    avg = float(np.mean(valid_elevs)) if valid_elevs else 291.0
    
    logger.debug(f"Elevation range: min={min(valid_elevs) if valid_elevs else 291.0:.2f}, max={max(valid_elevs) if valid_elevs else 291.0:.2f}, avg={avg:.2f}")
    return merged, avg

if __name__ == "__main__":
    points_us = [(38.9, -77.03), (39.1, -77.0)]
    points_br = [(-15.8, -47.9), (-23.5, -46.6)]
    elevs_us, avg_us = get_srtm_elevation_bulk(points_us)
    elevs_br, avg_br = get_srtm_elevation_bulk(points_br)
    print("US elevations:", elevs_us, "avg:", avg_us)
    print("Brazil elevations:", elevs_br, "avg:", avg_br)