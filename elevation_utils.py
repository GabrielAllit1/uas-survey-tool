import requests
import time
import logging
import random

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def _validate_and_format_points(points):
    """Ensure lat/lng are in correct order and within valid bounds."""
    formatted = []
    for idx, (lon, lat) in enumerate(points):
        # Try both (lon, lat) and (lat, lon)
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            formatted.append((lat, lon))
        elif -90 <= lon <= 90 and -180 <= lat <= 180:
            logger.warning("Swapping invalid coordinate pair #%d: (%s, %s)", idx, lon, lat)
            formatted.append((lon, lat))
        else:
            logger.error("Skipping invalid point: (%s, %s)", lon, lat)
    return formatted

def get_elevation_data(points, dataset='ned10m', batch_size=5, max_retries=5, method='GET', sleep_base=2, max_sleep=10):
    """
    Query OpenTopoData API in batches with retries and robust error handling.
    :param points: List of (lon, lat) tuples (will be validated/swapped as needed)
    :param dataset: Elevation dataset (default: 'ned10m')
    """
    points = _validate_and_format_points(points)
    elevations = []
    url = f"https://api.opentopodata.org/v1/{dataset}"
    session = requests.Session()

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        batch_results = [None] * len(batch)

        for attempt in range(1, max_retries + 1):
            try:
                if method.upper() == 'POST':
                    payload = {"locations": [{"latitude": lat, "longitude": lon} for lat, lon in batch]}
                    logger.debug("POSTing to %s: %s", url, payload)
                    response = session.post(url, json=payload, timeout=10)
                else:
                    location_str = "|".join(f"{lat},{lon}" for lat, lon in batch)
                    params = {"locations": location_str}
                    logger.debug("GETting from %s: %s", url, params)
                    response = session.get(url, params=params, timeout=10)

                if response.status_code == 429:
                    jitter = random.uniform(0.5, 1.5)
                    wait = min(sleep_base ** attempt * jitter, max_sleep)
                    logger.warning("Rate limited (429). Backing off for %.2f seconds...", wait)
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                data = response.json()
                if data.get("status") != "OK" or "results" not in data:
                    logger.error("Invalid response (status %s): %s", data.get("status"), data)
                    break

                batch_results = [res.get("elevation") for res in data["results"]]
                logger.debug("Batch %d-%d elevations: %s", i, i + len(batch), batch_results)
                break

            except requests.exceptions.RequestException as e:
                jitter = random.uniform(0.5, 1.5)
                wait = min(sleep_base ** attempt * jitter, max_sleep)
                logger.error("Request error (attempt %d): %s. Retrying in %.2f seconds...", attempt, str(e), wait)
                time.sleep(wait)
                if attempt == max_retries:
                    logger.warning("Giving up on batch %d-%d after %d retries.", i, i + len(batch), max_retries)

        elevations.extend(batch_results)

    session.close()
    return elevations
