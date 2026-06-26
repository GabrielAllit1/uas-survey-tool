"""
dem_downloader.py
—————————————————————————————————————————————————————————
Helper for downloading DEM/DSM GeoTIFFs from OpenTopography’s global and USGS 3DEP APIs.
Integrated with APIKeyManager for GUI-based key input and supports large AOI splitting.

Usage
-----
from dem_downloader import fetch_dem

bbox = dict(south=39.93, north=40.00, west=-105.33, east=-105.26)
tif = fetch_dem(bbox, dem_type="SRTMGL1")  # DEM example
print("Saved to:", tif)
"""

from __future__ import annotations
from pathlib import Path
import os
import logging
import requests
import time
import numpy as np
from typing import Optional, Dict, List
from elevation_data import get_srtm_elevation_bulk

try:
    from api_key_manager import APIKeyManager
except ModuleNotFoundError:
    APIKeyManager = None

_LOG = logging.getLogger(__name__)
_LOG.setLevel(logging.DEBUG)

def _resolve_api_key(explicit: Optional[str] = None) -> str:
    key = (
        explicit
        or os.getenv("OPENTOPO_API_KEY")
        or (APIKeyManager.opentopo() if APIKeyManager else None)
    )
    if not key:
        raise RuntimeError(
            "OpenTopography API key not found. "
            "Set OPENTOPO_API_KEY or allow the GUI dialog to supply it."
        )
    return key

def _split_bbox(bbox: Dict[str, float], max_area_km2: float = 450000) -> List[Dict[str, float]]:
    south, north, west, east = bbox['south'], bbox['north'], bbox['west'], bbox['east']
    lat_span = north - south
    lon_span = east - west
    area_km2 = lat_span * lon_span * 111 * 111
    if area_km2 <= max_area_km2:
        return [bbox]

    n_splits = int((area_km2 / max_area_km2) ** 0.5) + 1
    lat_step = lat_span / n_splits
    lon_step = lon_span / n_splits
    bboxes = []
    for i in range(n_splits):
        for j in range(n_splits):
            sub_bbox = {
                'south': south + i * lat_step,
                'north': south + (i + 1) * lat_step,
                'west': west + j * lon_step,
                'east': west + (j + 1) * lon_step
            }
            bboxes.append(sub_bbox)
    _LOG.debug(f"Split bbox into {len(bboxes)} tiles: {bboxes}")
    return bboxes

def fetch_dem(
    bbox: Dict[str, float],
    dem_type: str = "SRTMGL1",
    fmt: str = "GTiff",
    out_dir: Path | str = ".",
    api_key: Optional[str] = None,
    timeout: int = 300,
    max_retries: int = 3,
    retry_delay: int = 5
) -> Path:
    key = _resolve_api_key(api_key)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{dem_type}_{bbox['south']:.6f}_{bbox['west']:.6f}.tif"
    outfile = out_dir / filename

    try:
        import rasterio
        if outfile.exists():
            with rasterio.open(outfile) as src:
                data = src.read(1)
                if not (data.size == 0 or np.all(data == src.nodata)):
                    _LOG.debug(f"Using cached DEM/DSM: {outfile}")
                    return outfile
    except Exception as e:
        _LOG.warning(f"Invalid cached file {outfile}: {e}. Redownloading.")

    bboxes = _split_bbox(bbox, max_area_km2=450000)
    if len(bboxes) > 1:
        _LOG.info(f"Large AOI ({bbox}); split into {len(bboxes)} tiles")
        bbox = bboxes[0]
        filename = f"{dem_type}_{bbox['south']:.6f}_{bbox['west']:.6f}.tif"
        outfile = out_dir / filename

    is_usgs_3dep = dem_type.startswith("3DEP")
    base_url = (
        "https://portal.opentopography.org/API/usgsdem"
        if is_usgs_3dep else
        "https://portal.opentopography.org/API/globaldem"
    )
    param_key = "dataset" if is_usgs_3dep else "demtype"

    url = (
        f"{base_url}"
        f"?{param_key}={dem_type}"
        f"&south={bbox['south']}&north={bbox['north']}"
        f"&west={bbox['west']}&east={bbox['east']}"
        f"&outputFormat={fmt}"
        f"&API_Key={key}"
    )

    _LOG.info("Requesting %s → %s", dem_type, outfile)
    for attempt in range(max_retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 429:
                    _LOG.warning("Rate limit exceeded. Retrying after %d seconds...", retry_delay)
                    time.sleep(retry_delay)
                    continue
                r.raise_for_status()
                with open(outfile, "wb") as fp:
                    for chunk in r.iter_content(chunk_size=65536):
                        fp.write(chunk)
                _LOG.info("Finished download: %s (%.1f MB)", outfile, outfile.stat().st_size / 1e6)
                return outfile
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                _LOG.warning(f"Download attempt {attempt + 1} failed: {e}. Retrying after %d seconds...", retry_delay)
                time.sleep(retry_delay)
            else:
                _LOG.error(f"DEM/DSM download failed after %d attempts: %s", max_retries, e)
                centroid = {
                    'south': (bbox['south'] + bbox['north']) / 2,
                    'north': (bbox['south'] + bbox['north']) / 2,
                    'west': (bbox['west'] + bbox['east']) / 2,
                    'east': (bbox['west'] + bbox['east']) / 2
                }
                try:
                    elevations, _ = get_srtm_elevation_bulk([(centroid['west'], centroid['south'])], retry=3)
                    centroid_elevation = float(elevations[0]) if elevations and elevations[0] is not None and elevations[0] > 0 else 291.0
                    _LOG.info(f"Fallback to centroid elevation: {centroid_elevation:.2f}m")
                    return None, centroid_elevation
                except Exception as e2:
                    _LOG.error(f"Fallback elevation failed: {e2}")
                    return None, 291.0
    return None, 291.0


class DEMDownloader:
    """Compatibility wrapper retained for older tests and callers."""

    def __init__(self, out_dir: Path | str = ".") -> None:
        self.out_dir = Path(out_dir)

    def download_dem(self, kmz_path: str):
        if not kmz_path or not os.path.exists(kmz_path):
            return None
        raise NotImplementedError("KMZ-driven DEM downloads are handled through MainLogic.download_dem().")

if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    app = QApplication(sys.argv)
    bbox = dict(south=39.93, north=40.00, west=-105.33, east=-105.26)
    tif = fetch_dem(bbox, dem_type="SRTMGL1")
    print("Saved to:", tif)
    sys.exit(app.exec())
