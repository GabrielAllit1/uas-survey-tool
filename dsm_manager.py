import logging
import os
from typing import List, Tuple, Optional
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError
from pyproj import Transformer, CRS
from elevation_data import get_srtm_elevation_bulk
from api_key_manager import APIKeyManager
from PyQt6.QtCore import QThread, QMetaObject

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

SQM_PER_ACRE = 4046.856422

def _ensure_crs(crs_like) -> Optional[CRS]:
    try:
        if isinstance(crs_like, CRS):
            return crs_like
        if crs_like is None:
            return None
        return CRS.from_user_input(crs_like)
    except Exception as e:
        logger.error("Failed to parse CRS %r: %s", crs_like, e)
        return None

def _transform_points(points_lonlat: List[Tuple[float, float]], src_crs: CRS, dst_crs: CRS) -> List[Tuple[float, float]]:
    if src_crs == dst_crs:
        return points_lonlat
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return [transformer.transform(lon, lat) for lon, lat in points_lonlat]

def _sample_raster(points_lonlat: List[Tuple[float, float]], raster_path: str) -> List[float]:
    try:
        with rasterio.open(raster_path) as ds:
            ds_crs = _ensure_crs(ds.crs)
            if ds_crs is None:
                logger.error("Raster has no valid CRS: %s", raster_path)
                return [None] * len(points_lonlat)
            src_crs = CRS.from_epsg(4326)
            pts = points_lonlat
            if ds_crs != src_crs:
                pts = _transform_points(points_lonlat, src_crs, ds_crs)
            elevs = []
            for val in ds.sample(pts):
                v = float(val[0]) if len(val) else np.nan
                if ds.nodata is not None and v == ds.nodata:
                    v = np.nan
                if np.isnan(v) or v <= 0:
                    v = None
                elevs.append(v)
            logger.debug(f"Sampled {len(elevs)} elevations from raster {raster_path}")
            return elevs
    except RasterioIOError as e:
        logger.warning("Rasterio could not open %s: %s", raster_path, e)
        return [None] * len(points_lonlat)
    except Exception as e:
        logger.exception("Unexpected error during raster sampling: %s", e)
        return [None] * len(points_lonlat)

def get_dsm_elevation_profile(
    coords: List[Tuple[float, float]],
    raster_path: Optional[str] = None,
    utm_crs: Optional[CRS] = None,
    use_dsm: bool = True,
    pattern_type: str = 'grid',
    **kwargs
) -> Tuple[List[float], float, float]:
    try:
        points = []
        for item in coords:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                lon, lat = float(item[0]), float(item[1])
                if -180 <= lon <= 180 and -90 <= lat <= 90:
                    points.append((lon, lat))
                else:
                    logger.warning("Invalid coordinate (%f, %f) skipped", lon, lat)
            else:
                raise ValueError("Each coordinate must be (lon, lat)")

        if not points:
            logger.error("No valid coordinates provided")
            return [291.0] * len(coords), 0.5, 0.5

        # Accept aliases
        raster_path = raster_path or kwargs.get('dsm_path') or kwargs.get('dem_path') or kwargs.get('raster')

        # Raster sampling
        elevs = [None] * len(points)
        source = "none"
        if raster_path and os.path.exists(raster_path):
            elevs = _sample_raster(points, raster_path)
            source = "raster"

        # Fallback to OpenTopoData if insufficient valid data
        valid = [e for e in elevs if e is not None and e > 0]
        zero_frac = sum(1 for e in elevs if e == 0) / len(elevs) if elevs else 1.0
        if len(valid) < max(5, int(0.6 * len(elevs))) or zero_frac > 0.8:
            logger.debug("Raster sampling insufficient (valid=%d/%d, zero_frac=%.2f) → OpenTopoData fallback",
                         len(valid), len(elevs), zero_frac)
            try:
                elevs, _ = get_srtm_elevation_bulk(points, retry=3, dataset='srtm90m')
                source = "OpenTopoData"
                valid = [e for e in elevs if e is not None and e > 0]
                if len(valid) < max(5, int(0.6 * len(elevs))) or zero_frac > 0.8:
                    logger.debug("OpenTopoData elevations insufficient; trying alternative dataset")
                    elevs, _ = get_srtm_elevation_bulk(points, retry=3, dataset='srtm30m')
                    source = "OpenTopoData (srtm30m)"
            except Exception as e:
                logger.warning("OpenTopoData fallback failed: %s", e)
                elevs = [291.0] * len(points)
                source = "default"

        # Validate elevations
        valid_elevs = [e for e in elevs if e is not None and e > 0]
        if not valid_elevs:
            logger.warning("No valid elevations retrieved from %s; using default 291.0", source)
            elevs = [291.0] * len(points)
            source = "default"

        logger.info(f"Elevations retrieved from {source}: {len(valid_elevs)} valid elevations")

        # Scores
        try:
            from math_utils import unified_detection
            cycle_score = unified_detection(elevs, 64, 0.0)
            pattern_score = unified_detection(elevs, 32, 0.0)
        except Exception as e:
            logger.warning("unified_detection failed: %s, using fallback", e)
            valid_elevs_array = np.array(valid_elevs, dtype=float)
            if valid_elevs_array.size < 3:
                cycle_score = pattern_score = 0.5
            else:
                s_all = np.std(valid_elevs_array) + 1e-9
                s_d = np.std(np.diff(valid_elevs_array)) if valid_elevs_array.size >= 3 else 0.0
                cycle_score = pattern_score = float(np.clip(s_d / s_all, 0, 1))

        return [float(e) if e is not None else 291.0 for e in elevs], float(cycle_score), float(pattern_score)

    except Exception as e:
        logger.exception("Error in get_dsm_elevation_profile: %s", e)
        return [291.0] * len(coords), 0.5, 0.5

def suggest_layout_based_on_dsm(
    elevations: List[float],
    area_m2: Optional[float] = None,
    cycle_score: Optional[float] = None,
    pattern_score: Optional[float] = None
) -> str:
    try:
        arr = np.array([e for e in elevations if e is not None and e > 0], dtype=float)
        elev_range = float(arr.max() - arr.min()) if arr.size else 0.0
        if cycle_score is None:
            try:
                from math_utils import unified_detection
                cycle_score = unified_detection(arr, 64, 0.0)
            except Exception:
                cycle_score = 0.5
        if pattern_score is None:
            try:
                from math_utils import unified_detection
                pattern_score = unified_detection(arr, 32, 0.0)
            except Exception:
                pattern_score = 0.5
        complexity = (float(cycle_score) + float(pattern_score)) / 2.0
        acres = (float(area_m2) / SQM_PER_ACRE) if area_m2 else 0.0

        if elev_range > 50.0 or complexity > 0.7:
            return "fractal" if acres > 100.0 else "spiral"
        if elev_range > 20.0:
            return "triangular"
        return "grid"
    except Exception as e:
        logger.exception("Error in suggest_layout_based_on_dsm: %s", e)
        return "grid"