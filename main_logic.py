# main_logic.py
from __future__ import annotations

import logging
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import rasterio.mask
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from shapely.geometry import Point, Polygon, shape
from shapely.ops import transform as shapely_transform

from crs_utils import deduce_project_crs
from dem_downloader import fetch_dem
from dsm_manager import get_dsm_elevation_profile, suggest_layout_based_on_dsm
from elevation_data import get_srtm_elevation_bulk
from filter_coordinates import filter_coordinates_by_elevation, reject_coords_too_close
from flight_parameters_calculator import calculate_flight_parameters
from gcp_generator import generate_gcp
from proj_env import initialize_proj_env
from unified_mathtoolbox import MathToolBox

# Ensure PROJ is initialized (paths, etc.)
initialize_proj_env()

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
_LOG_DIR = os.path.expanduser("~/UAS_Survey_Tool_Logs")
os.makedirs(_LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(_LOG_DIR, 'main_logic.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Toolbox singleton
# -----------------------------------------------------------------------------
_toolbox = None
def get_toolbox():
    global _toolbox
    if _toolbox is None:
        # Back-compat: ignore unknown kwargs like device=...
        _toolbox = MathToolBox(use_cuda=False)
        logger.info("Math Tool Box initialized on device: cpu")
    return _toolbox


class MainLogic:
    """
    Main logic for UAS Survey Tool: loading AOI, computing flight params,
    generating GCP/VCP layouts, DSM/DEM handling, and diagnostics.
    """
    def __init__(self, cuda_enabled: bool = False, parent=None):
        self.cuda_enabled = bool(cuda_enabled)
        self.parent = parent
        self.logger = logging.getLogger(__name__)
        if self.cuda_enabled:
            try:
                import pycuda.driver as cuda  # noqa: F401
                self.logger.debug("MainLogic: CUDA requested; proceeding (lazy-init).")
            except Exception as e:
                self.logger.warning(f"CUDA not available; falling back to CPU: {e}")
                self.cuda_enabled = False

        # AOI / CRS
        self.polygon: Optional[Polygon] = None            # WGS84 polygon
        self.polygon_coords: List[Tuple[float, float]] = []
        self.polygon_proj: Optional[Polygon] = None       # Projected polygon (edit space)
        self.utm_crs: Optional[CRS] = None               # Working projected CRS (auto from centroid)
        self.crs: Optional[CRS] = None
        self.centroid: Optional[Tuple[float, float]] = None  # (lon, lat) in WGS84
        self.polygon_area_acres: Optional[float] = None
        self.centroid_elevation: float = 291.0

        # Survey / layout
        self.sequence_type: str = "fibonacci"
        self.modulus: Optional[int] = None
        self.layout_mode: str = "grid"
        self.survey_method: str = "LiDAR"
        self.min_gcp_count: int = 4
        self.relax_filters: bool = False

        # Paths
        self.last_kmz_path: Optional[str] = None
        self.cache_dir: str = os.path.join(tempfile.gettempdir(), "uas_survey_tool_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.dem_path: Optional[str] = None
        self.dsm_path: Optional[str] = None

        # Outputs
        self.gcp_points: List[dict] = []
        self.verification_points: List[dict] = []
        self.rejected_gcp_points: List[dict] = []
        self.rejected_verification_points: List[dict] = []

        # Flight params (filled by compute_flight_parameters)
        self.flight_params: Dict[str, Union[int, float, str]] = {
            'accuracy_mm': 60.96,
            'altitude_m': 80.0,
            'fov': 45.0,
            'prr': 640000.0,   # Hz
            'speed_mps': 10.0,
            'gsd_mm': 0.0,
            'ppsm': 0.0,
            'swath_width_m': 0.0,
            'image_width_px': 4000,
            'sensor_width_mm': 36.0,
            'focal_length_mm': 35.0,
            'layout_mode': 'grid',
        }

        self.logger.debug(f"Using cache directory: {self.cache_dir}")

    # -------------------------------------------------------------------------
    # CRS helpers
    # -------------------------------------------------------------------------
    def _ensure_projected_polygon(self):
        if self.polygon is None:
            return
        if self.crs is None:
            self.crs = deduce_project_crs(self.polygon_coords, fallback="EPSG:4326")
            self.utm_crs = self.crs
        if self.polygon_proj is None:
            tf = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
            self.polygon_proj = shapely_transform(lambda x, y: tf.transform(x, y), self.polygon)

    def lonlat_to_projected(self, lon: float, lat: float) -> Tuple[float, float]:
        if self.crs is None:
            self.crs = deduce_project_crs(self.polygon_coords, fallback="EPSG:4326")
            self.utm_crs = self.crs
        tf = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        return tf.transform(lon, lat)

    def projected_to_lonlat(self, e: float, n: float) -> Tuple[float, float]:
        if self.crs is None:
            self.crs = deduce_project_crs(self.polygon_coords, fallback="EPSG:4326")
            self.utm_crs = self.crs
        tf = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        return tf.transform(e, n)

    # -------------------------------------------------------------------------
    # KMZ loading
    # -------------------------------------------------------------------------
    def load_kmz(self, file_path: str) -> bool:
        """
        Load KMZ/KML → AOI polygon (WGS84), set centroid, area, and working CRS.
        """
        try:
            from kmz_parser import extract_geometries_from_kmz  # your existing parser
            geometries = extract_geometries_from_kmz(file_path)
            if not geometries:
                raise ValueError("No valid geometry found in file")

            geom = geometries[0]['geometry']  # shapely geometry in WGS84
            coords = geometries[0]['coords']
            if not coords or len(coords) < 3:
                raise ValueError("Invalid or empty coordinate list from KMZ/KML")
            for lon, lat in coords:
                if not (-180.0 <= float(lon) <= 180.0 and -90.0 <= float(lat) <= 90.0):
                    raise ValueError(f"Invalid coordinate: ({lon}, {lat})")

            self.last_kmz_path = file_path
            self.polygon = geom if isinstance(geom, Polygon) else shape(geom)
            self.polygon_coords = coords
            self.centroid = (self.polygon.centroid.x, self.polygon.centroid.y)

            # Auto-CRS from centroid, no hardcoding
            self.crs = deduce_project_crs(self.polygon_coords, fallback="EPSG:4326")
            self.utm_crs = self.crs
            self._ensure_projected_polygon()

            # Area in acres (project)
            gdf = gpd.GeoSeries([self.polygon], crs="EPSG:4326")
            gdf_utm = gdf.to_crs(self.crs)
            self.polygon_area_acres = float(gdf_utm.geometry.iloc[0].area) / 4046.86

            # Centroid elevation
            try:
                elevations, _ = get_srtm_elevation_bulk([(self.centroid[1], self.centroid[0])], retry=3)
                if elevations and elevations[0] and elevations[0] > 0:
                    self.centroid_elevation = float(elevations[0])
            except Exception as e:
                self.logger.warning(f"Failed to fetch centroid elevation: {e}")

            self.logger.info(
                f"AOI loaded. Centroid={self.centroid[0]:.6f},{self.centroid[1]:.6f}, "
                f"Area={self.polygon_area_acres:.2f} acres, CRS={self.crs.to_string()}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to load KMZ: {e}")
            return False

    # -------------------------------------------------------------------------
    # DEM/DSM downloads
    # -------------------------------------------------------------------------
    def download_dem(self, kmz_path: str):
        try:
            min_lon = min(lon for lon, _ in self.polygon_coords)
            max_lon = max(lon for lon, _ in self.polygon_coords)
            min_lat = min(lat for _, lat in self.polygon_coords)
            max_lat = max(lat for _, lat in self.polygon_coords)
            bbox = {'south': min_lat, 'north': max_lat, 'west': min_lon, 'east': max_lon}
            dem_path = fetch_dem(bbox, dem_type="SRTMGL1", out_dir=self.cache_dir)
            self.dem_path = str(dem_path) if dem_path else None
            self.logger.debug(f"Downloaded DEM to {self.dem_path}")
            return self.dem_path, self.centroid_elevation
        except Exception as e:
            self.logger.error(f"DEM download failed: {e}")
            return None, self.centroid_elevation

    def download_dsm(self, kmz_path: str):
        try:
            min_lon = min(lon for lon, _ in self.polygon_coords)
            max_lon = max(lon for lon, _ in self.polygon_coords)
            min_lat = min(lat for _, lat in self.polygon_coords)
            max_lat = max(lat for _, lat in self.polygon_coords)
            bbox = {'south': min_lat, 'north': max_lat, 'west': min_lon, 'east': max_lon}
            dsm_path = fetch_dem(bbox, dem_type="COP30", out_dir=self.cache_dir)
            self.dsm_path = str(dsm_path) if dsm_path else None
            self.logger.debug(f"Downloaded DSM to {self.dsm_path}")
            return self.dsm_path, self.centroid_elevation
        except Exception as e:
            self.logger.error(f"DSM download failed: {e}")
            return None, self.centroid_elevation

    # -------------------------------------------------------------------------
    # Flight parameters
    # -------------------------------------------------------------------------
    def compute_flight_parameters(
        self,
        accuracy_mm: float,
        altitude_m: float,
        fov_deg: float,
        prr_hz: float,
        *,
        focal_length_mm: float = 35.0,
        image_width_px: int = 4000,
        speed_mps: float = 10.0,
        swath_width_m: Optional[float] = None,
        sensor_width_mm: float = 36.0,
    ) -> Dict[str, float]:
        prr = float(prr_hz)
        if prr < 2000:
            prr *= 1000.0  # tolerate kHz inputs
        if swath_width_m is None or swath_width_m <= 0:
            swath_width_m = 2.0 * float(altitude_m) * float(np.tan(np.radians(fov_deg / 2.0)))
        params: Dict[str, float] = {}
        try:
            params = calculate_flight_parameters(
                accuracy_mm=accuracy_mm,
                altitude_m=altitude_m,
                fov_deg=fov_deg,
                prr_hz=prr,
                focal_length_mm=focal_length_mm,
                image_width_px=image_width_px,
                speed_mps=speed_mps,
                swath_width_m=swath_width_m,
                sensor_width_mm=sensor_width_mm,
            )
        except TypeError:
            altitude_m = float(altitude_m)
            fov_deg = float(fov_deg)
            speed_mps = float(speed_mps)
            focal_length_mm = float(focal_length_mm)
            image_width_px = int(image_width_px)
            sensor_width_mm = float(sensor_width_mm)
            gsd_m = (altitude_m * (sensor_width_mm * 1e-3)) / (focal_length_mm * image_width_px)
            gsd_mm = gsd_m * 1000.0
            ppsm = prr / max(swath_width_m * speed_mps, 1e-6)
            layout_k = {'grid': 50.0, 'triangular': 45.0, 'diamond': 48.0, 'spiral': 40.0, 'fractal': 35.0, 'chaotic': 30.0}.get(self.layout_mode, 50.0)
            required_accuracy_m = float(accuracy_mm) / 1000.0
            spacing_m = max(10.0, (altitude_m / (required_accuracy_m + 1e-6)) / layout_k)
            area_m2 = float(self.polygon_area_acres or 0.0) * 4046.86
            estimated_gcp_count = max(self.min_gcp_count, int(math.ceil(area_m2 / max(spacing_m**2, 1e-6))))
            params = {
                'accuracy_mm': float(accuracy_mm),
                'altitude_m': altitude_m,
                'fov': fov_deg,
                'prr': prr,
                'speed_mps': speed_mps,
                'swath_width_m': float(swath_width_m),
                'focal_length_mm': focal_length_mm,
                'image_width_px': image_width_px,
                'sensor_width_mm': sensor_width_mm,
                'gsd_mm': float(gsd_mm),
                'ppsm': float(ppsm),
                'estimated_gcp_count': float(estimated_gcp_count),
                'layout_mode': self.layout_mode,
            }
        self.flight_params.update(params)
        self.flight_params['layout_mode'] = self.layout_mode
        return self.flight_params

    # -------------------------------------------------------------------------
    # Layout & spacing helpers
    # -------------------------------------------------------------------------
    def set_survey_method(self, method: str, modulus: Optional[int] = None, weights: Optional[List[float]] = None):
        valid_methods = {"grid", "triangular", "diamond", "spiral", "chaotic", "fractal"}
        m = method.lower()
        if m == "lidar":
            self.survey_method = "LiDAR"
            self.layout_mode = "grid"
        elif m == "photogrammetry":
            self.survey_method = "Photogrammetry"
            self.layout_mode = "fractal"
        elif m in valid_methods:
            self.survey_method = method
            self.layout_mode = method
        else:
            raise ValueError(f"Survey method must be one of {valid_methods}, 'LiDAR', or 'Photogrammetry'")
        if modulus is not None:
            self.modulus = int(modulus)
        if weights is not None:
            if len(weights) != 4 or any(w < 0 for w in weights):
                self.logger.warning(f"Invalid weights {weights}; using default [0.25, 0.25, 0.25, 0.25]")
                weights = [0.25, 0.25, 0.25, 0.25]
            self.weights = weights

    def calculate_modulus(self, area_acres: float, estimated_gcp_count: int) -> int:
        fib = [get_toolbox().fibonacci(i) for i in range(20)]
        idx = min(max(int(np.log((area_acres or 0) + 1) * 2), 2), len(fib) - 1)
        modulus = int(fib[idx])
        while np.gcd(modulus, estimated_gcp_count) != 1 and idx < len(fib) - 1:
            idx += 1
            modulus = int(fib[idx])
        return modulus

    def calculate_recommended_spacing_from_payload(self) -> float:
        try:
            area_acres = float(self.polygon_area_acres or 1039.95)
            area_m2 = area_acres * 4046.86
            altitude_m = float(self.flight_params.get('altitude_m', 80.0))
            fov_deg = float(self.flight_params.get('fov', 45.0))
            prr = float(self.flight_params.get('prr', 640000.0))
            accuracy_mm = float(self.flight_params.get('accuracy_mm', 60.96))
            speed_mps = float(self.flight_params.get('speed_mps', 10.0))
            swath_width = 2 * altitude_m * np.tan(np.radians(fov_deg / 2))
            ppsm = prr / max(swath_width * speed_mps, 1e-6)
            sensor_width_mm = float(self.flight_params.get('sensor_width_mm', 36.0))
            focal_length_mm = float(self.flight_params.get('focal_length_mm', 35.0))
            image_width_px = int(self.flight_params.get('image_width_px', 4000))
            gsd_m = (altitude_m * (sensor_width_mm * 1e-3)) / (focal_length_mm * image_width_px)
            gsd_mm = gsd_m * 1000.0
            k = {'grid': 50.0, 'triangular': 45.0, 'diamond': 48.0, 'spiral': 40.0, 'fractal': 35.0, 'chaotic': 30.0}.get(self.layout_mode, 50.0)
            terrain_factor = 1.0 + (10.0 / 50.0)
            required_accuracy_m = float(accuracy_mm) / 1000.0
            spacing = max(10.0, np.sqrt(area_m2) / (k * required_accuracy_m * terrain_factor))
            fib = [get_toolbox().fibonacci(i) for i in range(20)]
            index = min(max(int(np.log(area_acres + 1) * 2), 2), len(fib) - 1)
            fam_scale = min(fib[index] / (fib[5] if fib[5] != 0 else 1.0), 2.0)
            spacing *= fam_scale
            self.logger.debug(f"Calculated recommended spacing: {spacing:.2f}m (GSD={gsd_mm:.2f}mm, PPSM={ppsm:.2f})")
            return float(spacing)
        except Exception as e:
            self.logger.error(f"Failed to calculate recommended spacing: {e}")
            return 300.0

    # -------------------------------------------------------------------------
    # Elevation helpers
    # -------------------------------------------------------------------------
    def sample_elevation_at(self, lon: float, lat: float) -> float:
        """
        Sample DSM/DEM at lon/lat; fallback to OpenTopoData; fallback to centroid_elevation.
        """
        try:
            if self.dsm_path and os.path.exists(self.dsm_path):
                with rasterio.open(self.dsm_path) as ds:
                    row, col = ds.index(lon, lat)
                    if 0 <= row < ds.height and 0 <= col < ds.width:
                        val = list(ds.sample([(lon, lat)]))[0][0]
                        if val is not None and np.isfinite(val):
                            return float(val)
            if self.dem_path and os.path.exists(self.dem_path):
                with rasterio.open(self.dem_path) as ds:
                    row, col = ds.index(lon, lat)
                    if 0 <= row < ds.height and 0 <= col < ds.width:
                        val = list(ds.sample([(lon, lat)]))[0][0]
                        if val is not None and np.isfinite(val):
                            return float(val)
        except Exception as e:
            self.logger.debug(f"Direct raster sample failed at ({lon},{lat}): {e}")

        try:
            elevs, _ = get_srtm_elevation_bulk([(lat, lon)], retry=2)
            if elevs and elevs[0] and elevs[0] > 0:
                return float(elevs[0])
        except Exception:
            pass
        return float(self.centroid_elevation)

    def get_dsm_elevations_or_fallback(self, latlon_points: List[Tuple[float, float]], dsm_path: Optional[str]) -> Tuple[List[float], float, float]:
        try:
            if dsm_path and os.path.exists(dsm_path):
                # IMPORTANT: dsm_manager expects (lon, lat) tuples
                elevs, cycle_score, pattern_score = get_dsm_elevation_profile(latlon_points, dsm_path)
                vals = [float(e) for e in elevs if e is not None and e > 0]
                if vals:
                    return vals, float(cycle_score), float(pattern_score)
        except Exception as e:
            self.logger.warning(f"DSM sampling failed: {e}")
        try:
            elevs, _ = get_srtm_elevation_bulk(latlon_points, retry=3)
            vals = [float(e) for e in elevs if e is not None and e > 0]
            cycle_score = 0.0
            pattern_score = float(get_toolbox().unified_detection(len(vals) or 1, 0))
            return (vals if vals else [self.centroid_elevation] * len(latlon_points)), cycle_score, pattern_score
        except Exception as e:
            self.logger.warning(f"SRTM/OpenTopo fallback failed: {e}")
            return [self.centroid_elevation] * len(latlon_points), 0.0, 0.0

    # -------------------------------------------------------------------------
    # Suitability check
    # -------------------------------------------------------------------------
    def is_suitable_location(self, px: float, py: float, dem_dataset, dsm_dataset, transformer_to_wgs84: Transformer):
        self.logger.debug(f"Entering is_suitable_location for point ({px:.2f}, {py:.2f})")
        if dem_dataset is None and dsm_dataset is None:
            return True, "Suitable (no DEM/DSM)", self.centroid_elevation
        try:
            lon, lat = transformer_to_wgs84.transform(px, py)
            ds = dsm_dataset or dem_dataset
            row, col = ds.index(lon, lat)
            if not (0 <= row < ds.height and 0 <= col < ds.width):
                return False, "Outside DEM/DSM bounds", None
            value = list(ds.sample([(lon, lat)]))[0][0]
            if value == ds.nodata or not np.isfinite(value):
                return False, "Invalid elevation", None
            return True, "Suitable", float(value)
        except Exception as e:
            self.logger.debug(f"Suitability check failed: {e}")
            return True, "Suitable (check failed; allowing)", self.centroid_elevation

    # -------------------------------------------------------------------------
    # Reprojection patterns (fast, non-destructive)
    # -------------------------------------------------------------------------
    def _generate_candidates(self, spacing_m: float, pattern: str) -> List[Tuple[float, float]]:
        """
        Generate candidate (easting, northing) points in projected CRS inside AOI.
        """
        self._ensure_projected_polygon()
        poly = self.polygon_proj
        if poly is None or poly.is_empty:
            return []
        minx, miny, maxx, maxy = poly.bounds
        width = maxx - minx
        height = maxy - miny
        cx, cy = poly.centroid.x, poly.centroid.y

        pts: List[Tuple[float, float]] = []
        pattern = (pattern or "grid").lower()

        if pattern == "grid" or pattern == "diamond":
            # Base grid
            dx = dy = max(spacing_m, 1.0)
            x_vals = np.arange(minx, maxx + dx, dx)
            y_vals = np.arange(miny, maxy + dy, dy)
            for yi, y in enumerate(y_vals):
                for xi, x in enumerate(x_vals):
                    xx, yy = x, y
                    if pattern == "diamond":
                        # rotate 45° around centroid
                        ang = math.pi / 4.0
                        rx = cx + (xx - cx) * math.cos(ang) - (yy - cy) * math.sin(ang)
                        ry = cy + (xx - cx) * math.sin(ang) + (yy - cy) * math.cos(ang)
                        xx, yy = rx, ry
                    p = Point(xx, yy)
                    if poly.contains(p):
                        pts.append((xx, yy))

        elif pattern == "triangular":
            dx = max(spacing_m, 1.0)
            dy = dx * math.sin(math.pi/3.0)
            x_vals = np.arange(minx, maxx + dx, dx)
            y_vals = np.arange(miny, maxy + dy, dy)
            for j, y in enumerate(y_vals):
                offset = 0.5 * dx if (j % 2) else 0.0
                for x in (x_vals + offset):
                    p = Point(x, y)
                    if poly.contains(p):
                        pts.append((x, y))

        elif pattern == "spiral":
            # Archimedean spiral from centroid
            a = 0.0
            b = max(spacing_m, 1.0) / (2 * math.pi)
            r_max = 0.75 * max(width, height)
            t = 0.0
            while True:
                r = a + b * t
                if r > r_max:
                    break
                x = cx + r * math.cos(t)
                y = cy + r * math.sin(t)
                if poly.contains(Point(x, y)):
                    pts.append((x, y))
                t += 0.45  # angle step

        elif pattern == "chaotic":
            # Jittered grid seeds
            dx = dy = max(spacing_m, 1.0)
            x_vals = np.arange(minx, maxx + dx, dx)
            y_vals = np.arange(miny, maxy + dy, dy)
            rng = random.Random(42)
            for y in y_vals:
                for x in x_vals:
                    jx = x + rng.uniform(-0.35*dx, 0.35*dx)
                    jy = y + rng.uniform(-0.35*dy, 0.35*dy)
                    if poly.contains(Point(jx, jy)):
                        pts.append((jx, jy))

        elif pattern == "fractal":
            # Multiscale grid union (coarse + medium + fine)
            for scale in (1.6, 1.0, 0.7):
                dx = dy = max(spacing_m * scale, 1.0)
                x_vals = np.arange(minx, maxx + dx, dx)
                y_vals = np.arange(miny, maxy + dy, dy)
                for y in y_vals:
                    for x in x_vals:
                        if poly.contains(Point(x, y)):
                            pts.append((x, y))
        else:
            # Fallback to grid
            return self._generate_candidates(spacing_m, "grid")

        return pts

    def reproject_points_to_pattern(
        self,
        gcp_points: List[dict],
        vcp_points: List[dict],
        *,
        layout_mode: str,
        spacing_m: float,
        sequence_type: str = "fibonacci",
        min_distance_m: float = 0.2
    ) -> Tuple[List[dict], List[dict]]:
        """
        Reproject current points into a chosen pattern, inside AOI, in projected CRS.
        Keeps total counts and GCP/VCP proportions, assigns elevations, and returns
        updated dicts with lon/lat/easting/northing.
        """
        self._ensure_projected_polygon()
        poly = self.polygon_proj
        if poly is None or poly.is_empty:
            return gcp_points, vcp_points

        tgt_count = max(1, len(gcp_points) + len(vcp_points))
        gcp_ratio = (len(gcp_points) / tgt_count) if tgt_count else 0.5
        tgt_gcp = max(1, int(round(tgt_count * gcp_ratio)))
        tgt_vcp = max(0, tgt_count - tgt_gcp)

        # Generate many candidates and then filter down
        cands = self._generate_candidates(spacing_m, layout_mode)
        if not cands:
            return gcp_points, vcp_points

        # Simple min-distance thinning
        sel: List[Tuple[float, float]] = []
        md2 = max(min_distance_m, 0.05) ** 2
        for (x, y) in cands:
            ok = True
            for (sx, sy) in sel:
                if (x - sx) * (x - sx) + (y - sy) * (y - sy) < md2:
                    ok = False
                    break
            if ok:
                sel.append((x, y))
            if len(sel) >= tgt_count * 3:  # pool big enough
                break
        if not sel:
            sel = cands[:tgt_count]

        # Convert to lon/lat & build dicts
        out: List[dict] = []
        for i, (e, n) in enumerate(sel):
            lon, lat = self.projected_to_lonlat(e, n)
            out.append({
                "name": f"P{i+1}",
                "easting": float(e),
                "northing": float(n),
                "lon": float(lon),
                "lat": float(lat),
                "elevation": float(self.sample_elevation_at(lon, lat))
            })

        # Stable shuffle based on requested sequence type
        rng = random.Random(123 if sequence_type.lower().startswith("fib") else 321)
        rng.shuffle(out)

        new_gcps = []
        new_vcps = []
        for i, p in enumerate(out[:tgt_gcp]):
            q = dict(p)
            q["name"] = f"GCP_{i+1}"
            q["type"] = "GCP"
            new_gcps.append(q)
        for i, p in enumerate(out[tgt_gcp:tgt_gcp+tgt_vcp]):
            q = dict(p)
            q["name"] = f"VCP_{i+1}"
            q["type"] = "VCP"
            new_vcps.append(q)

        return new_gcps, new_vcps

    # -------------------------------------------------------------------------
    # Core plan generation (original path)
    # -------------------------------------------------------------------------
    def generate_gcp_plan(
        self,
        *,
        auto_download_dem: bool,
        auto_download_dsm: bool,
        user_spacing: Optional[float],
        layout_mode: str,
        sequence_type: str,
        weights: Optional[List[float]],
        dem_path: Optional[str],
        dsm_path: Optional[str]
    ):
        if not self.crs:
            self.crs = deduce_project_crs(self.polygon_coords, fallback="EPSG:4326")
            self.utm_crs = self.crs
        self._ensure_projected_polygon()

        self.layout_mode = layout_mode or self.layout_mode
        self.sequence_type = sequence_type or self.sequence_type
        if weights is not None:
            if len(weights) != 4 or any(w < 0 for w in weights):
                self.logger.warning(f"Invalid weights {weights}; using default [0.25,0.25,0.25,0.25]")
                weights = [0.25, 0.25, 0.25, 0.25]
            self.weights = weights

        if auto_download_dem and not dem_path:
            self.dem_path, _ = self.download_dem(self.last_kmz_path or "")
        else:
            self.dem_path = dem_path
        if auto_download_dsm and not dsm_path:
            self.dsm_path, _ = self.download_dsm(self.last_kmz_path or "")
        else:
            self.dsm_path = dsm_path

        spacing = float(user_spacing) if user_spacing else self.calculate_recommended_spacing_from_payload()

        fib_seq = [get_toolbox().fibonacci(i) for i in range(1001)]
        kap_seq = [0.5] * 1000

        gcp_points, verification_points, rejected_g, rejected_v = generate_gcp(
            geometry_wgs84=self.polygon,
            spacing=spacing,
            layout_mode=self.layout_mode,
            sequence_type=self.sequence_type,
            modulus=self.calculate_modulus(self.polygon_area_acres or 1.0, 25),
            min_distance_factor=0.65,
            dem_path=self.dem_path,
            dsm_path=self.dsm_path,
            fib_sequence=fib_seq,
            kap_sequence=kap_seq,
            main_logic=self,
            density_factor=1.0,
            gcp_density=spacing,
            manual_override=False
        )
        self.gcp_points = gcp_points or []
        self.verification_points = verification_points or []
        self.rejected_gcp_points = rejected_g or []
        self.rejected_verification_points = rejected_v or []

        # IMPORTANT: dsm_manager.get_dsm_elevation_profile expects (lon, lat) tuples
        lonlat_points = [(p['lon'], p['lat']) for p in (self.gcp_points + self.verification_points)]
        elevs, cycle_score, pattern_score = self.get_dsm_elevations_or_fallback(lonlat_points, self.dsm_path or self.dem_path)
        elevs = [float(e) for e in elevs if e is not None and np.isfinite(e)]
        if not elevs:
            elevs = [self.centroid_elevation] * max(1, len(lonlat_points))

        suggested_layout = suggest_layout_based_on_dsm(
            elevs,
            (self.polygon_area_acres or 0.0) * 4046.86,
            cycle_score,
            pattern_score
        )

        elevation_data = {
            "profile": elevs,
            "min": float(np.min(elevs)),
            "max": float(np.max(elevs)),
            "mean": float(np.mean(elevs)),
            "cycle_score": float(cycle_score),
            "pattern_score": float(pattern_score),
            "suggested_layout": suggested_layout,
            "recommended_spacing_m": spacing
        }

        return self.gcp_points, self.verification_points, elevation_data

    # -------------------------------------------------------------------------
    # Optional: 3D model
    # -------------------------------------------------------------------------
    def generate_3d_model(self, elevations_profile, polygon_coords, out_dir=None):
        import numpy as np, os, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        def _bounds_ll(coords):
            if not coords: return None
            lons = [c[0] for c in coords]; lats = [c[1] for c in coords]
            return min(lons), min(lats), max(lons), max(lats)
        try:
            out_dir = out_dir or os.path.join(self.cache_dir, "renders")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "terrain_3d.png")
            grid_Z = None; X = None; Y = None
            tif_path = self.dsm_path if (self.dsm_path and os.path.exists(self.dsm_path)) else (self.dem_path if (self.dem_path and os.path.exists(self.dem_path)) else None)
            if tif_path:
                try:
                    with rasterio.open(tif_path) as ds:
                        if polygon_coords:
                            min_lon, min_lat, max_lon, max_lat = _bounds_ll(polygon_coords)
                        else:
                            b = ds.bounds
                            min_lon, min_lat, max_lon, max_lat = b.left, b.bottom, b.right, b.top
                        nx = ny = 80
                        xs = np.linspace(min_lon, max_lon, nx)
                        ys = np.linspace(min_lat, max_lat, ny)
                        Z = np.zeros((ny, nx), dtype=float)
                        for j, yy in enumerate(ys):
                            rows = list(ds.sample([(xx, yy) for xx in xs]))
                            Z[j, :] = [float(v[0]) if v[0] != ds.nodata and v[0] == v[0] else np.nan for v in rows]
                        crs_obj = deduce_project_crs(polygon_coords, fallback="EPSG:4326")
                        tf = Transformer.from_crs("EPSG:4326", crs_obj, always_xy=True)
                        cen_lon = float((min_lon + max_lon) / 2.0); cen_lat = float((min_lat + max_lat) / 2.0)
                        exs, nys = [], []
                        for xx in xs:
                            ex, _ = tf.transform(xx, cen_lat); exs.append(ex)
                        for yy in ys:
                            _, nyv = tf.transform(cen_lon, yy); nys.append(nyv)
                        X, Y = np.meshgrid(np.array(exs), np.array(nys))
                        grid_Z = Z
                except Exception as e:
                    self.logger.warning(f"3D model DSM/DEM sampling failed: {e}")
            if grid_Z is None:
                try:
                    if not polygon_coords or len(polygon_coords) < 3:
                        return None
                    min_lon, min_lat, max_lon, max_lat = _bounds_ll(polygon_coords)
                    nx = ny = 80
                    xs = np.linspace(min_lon, max_lon, nx)
                    ys = np.linspace(min_lat, max_lat, ny)
                    query_pts = [(float(lat), float(lon)) for lat in ys for lon in xs]
                    elevs, _ = get_srtm_elevation_bulk(query_pts, retry=3)
                    arr = np.array([float(v) if v is not None else np.nan for v in elevs], dtype=float)
                    grid_Z = arr.reshape((ny, nx))
                    crs_obj = deduce_project_crs(polygon_coords, fallback="EPSG:4326")
                    tf = Transformer.from_crs("EPSG:4326", crs_obj, always_xy=True)
                    cen_lon = float((min_lon + max_lon) / 2.0); cen_lat = float((min_lat + max_lat) / 2.0)
                    exs, nys = [], []
                    for xx in xs:
                        ex, _ = tf.transform(xx, cen_lat); exs.append(ex)
                    for yy in ys:
                        _, nyv = tf.transform(cen_lon, yy); nys.append(nyv)
                    X, Y = np.meshgrid(np.array(exs), np.array(nys))
                except Exception as e:
                    self.logger.warning(f"3D model OpenTopoData fallback failed: {e}")
                    return None
            if grid_Z is None or X is None or Y is None:
                return None
            Z = np.array(grid_Z, dtype=float)
            if np.isnan(Z).any():
                m = np.nanmean(Z)
                Z = np.where(np.isnan(Z), m, Z)
            fig = plt.figure(figsize=(7.5, 5.5))
            ax3 = fig.add_subplot(111, projection='3d')
            ax3.plot_surface(X, Y, Z, cmap='terrain', linewidth=0, antialiased=True)
            ax3.set_xlabel('Easting (m)'); ax3.set_ylabel('Northing (m)'); ax3.set_zlabel('Elevation (m)')
            plt.title('3D Terrain Model')
            plt.tight_layout()
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            self.logger.debug(f"3D model rendered to {out_path}")
            return out_path
        except Exception as e:
            self.logger.warning(f"generate_3d_model failed: {e}")
            return None
