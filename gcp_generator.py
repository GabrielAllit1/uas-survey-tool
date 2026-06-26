from functools import lru_cache
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING, Union
import math
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import transform as shapely_transform
from shapely.validation import make_valid
from shapely.coords import CoordinateSequence
from pyproj import CRS, Transformer
import rasterio
import logging
import random
import os
from scipy.spatial import cKDTree
from unified_mathtoolbox import MathToolBox
from elevation_data import get_srtm_elevation_bulk
from area_utils import calculate_area_acres
from elevation_service import detect_elevation_cycles
from filter_coordinates import reject_coords_too_close, filter_coordinates_by_elevation
from flight_parameters_calculator import calculate_flight_parameters
from proj_env import initialize_proj_env

if TYPE_CHECKING:
    from main_logic import MainLogic

initialize_proj_env()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.expanduser("~/UAS_Survey_Tool_Logs"), 'gcp_generator.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    import pycuda.driver as cuda
    cuda.init()
    device = cuda.Device(0)
    logger.info(f"CUDA detected: Using device '{device.name()}'")
    USE_CUDA = True
except Exception as e:
    logger.warning(f"CUDA initialization failed or unavailable, falling back to CPU-only mode: {e}")
    USE_CUDA = False

# Singleton-like MathToolBox instance
_toolbox = None
def get_toolbox():
    global _toolbox
    if _toolbox is None:
        _toolbox = MathToolBox()
        logger.debug("Initialized singleton MathToolBox")
    return _toolbox

@lru_cache(maxsize=1000)
def get_cached_elevation(lat: float, lon: float, dataset_path: str, dataset_type: str, default_elevation: float = 291.0) -> float:
    try:
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            logger.error(f"Invalid coordinates in get_cached_elevation: (lat={lat:.6f}, lon={lon:.6f})")
            return default_elevation
        if not dataset_path or not os.path.exists(dataset_path):
            logger.debug(f"No {dataset_type} dataset available, using default elevation")
            return default_elevation
        with rasterio.open(dataset_path) as dataset:
            row, col = dataset.index(lon, lat)
            if not (0 <= row < dataset.height and 0 <= col < dataset.width):
                logger.debug(f"Coordinates (lat={lat:.6f}, lon={lon:.6f}) outside {dataset_type} bounds")
                return default_elevation
            value = dataset.read(1)[row, col]
            return value if value != dataset.nodata else default_elevation
    except Exception as e:
        logger.warning(f"Failed to sample {dataset_type} at (lat={lat:.6f}, lon={lon:.6f}): {e}")
        return default_elevation

def calculate_fam_spacing(area_acres, base_spacing):
    toolbox = get_toolbox()
    fib = [toolbox.fibonacci(i) for i in range(21)]
    index = int(np.log(area_acres + 1) * 2)
    index = min(max(index, 2), len(fib)-1)
    return base_spacing * fib[index] / fib[5] if fib[5] != 0 else base_spacing

def calculate_gcp_spacing(area_acres, flight_params, area_m2):
    logger.debug(f"Calculating GCP spacing for area={area_acres:.2f} acres")
    required_keys = ['agl', 'sensor_width_mm', 'focal_length_mm', 'image_width_px', 'speed', 'swath_width_m', 'layout_mode', 'accuracy_mm']
    missing_keys = [key for key in required_keys if key not in flight_params]
    if missing_keys:
        logger.error(f"Missing required flight parameters: {missing_keys}")
        return 300.0, 47  # Default to 300m, ~47 GCPs for 1039.95 acres
    try:
        params = calculate_flight_parameters(
            area_m2=area_m2,
            altitude_m=flight_params['agl'],
            sensor_width_mm=flight_params['sensor_width_mm'],
            focal_length_mm=flight_params['focal_length_mm'],
            image_width_px=flight_params['image_width_px'],
            speed_mps=flight_params['speed'],
            swath_width_m=flight_params['swath_width_m'],
            layout_mode=flight_params['layout_mode'],
            desired_rmse_m=flight_params['accuracy_mm'] / 1000,
            terrain_elevation_variance=flight_params.get('terrain_elevation_variance', 10.0)
        )
        if params is None:
            logger.error("Flight parameter calculation failed, using default values")
            return 300.0, 47
        spacing = params['recommended_spacing_m']
        n_gcps = math.ceil(area_m2 / (spacing ** 2))
        logger.debug(f"Calculated {n_gcps} GCPs, spacing={spacing:.2f}m")
        return spacing, n_gcps
    except Exception as e:
        logger.error(f"Failed to calculate GCP spacing: {e}")
        return 300.0, 47

def get_utm_crs_for_geometry(coords: Union[List[Tuple[float, float]], CoordinateSequence]):
    """Determine the UTM zone from coordinates, handling CoordinateSequence."""
    try:
        logger.debug(f"Input coordinates type: {type(coords)}")
        if isinstance(coords, CoordinateSequence):
            coords = [(float(x), float(y)) for x, y in coords]
        valid_coords = [(lon, lat) for lon, lat in coords if -180 <= lon <= 180 and -90 <= lat <= 90]
        if not valid_coords or len(valid_coords) < 2:
            logger.error(f"No valid or insufficient coordinates provided: {valid_coords}")
            raise ValueError(f"No valid or insufficient coordinates provided: {valid_coords}")
        avg_lat = sum(lat for _, lat in valid_coords) / len(valid_coords)
        avg_lon = sum(lon for lon, _ in valid_coords) / len(valid_coords)
        zone = int((avg_lon + 180) / 6) + 1
        hemisphere = 'north' if avg_lat >= 0 else 'south'
        utm_crs = CRS.from_dict({'proj': 'utm', 'zone': zone, 'south': hemisphere == 'south'})
        logger.debug(f"Calculated UTM CRS: {utm_crs} for avg_lat={avg_lat:.2f}, avg_lon={avg_lon:.2f}")
        return utm_crs
    except Exception as e:
        logger.error(f"UTM zone calculation failed: {e}")
        return CRS.from_epsg(4326)  # Fallback to WGS84

def generate_fractal_gcp_layout(polygon, spacing, area_acres, expected_gcp_count):
    """Generate GCPs using fractal-like Fibonacci placement."""
    toolbox = get_toolbox()
    fib = [toolbox.fibonacci(i) for i in range(21)]
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    max_points = min(10000, int(expected_gcp_count * 1.5))  # Allow 50% extra for filtering
    for k in range(3, min(10, len(fib))):
        fk = fib[k]
        scale = spacing * fk / fib[5] if fib[5] != 0 else spacing
        for i in range(min(int(area_acres / fk), max_points - len(points))):
            x = minx + (maxx - minx) * (i % fk) / fk
            y = miny + (maxy - miny) * (i // fk) / fk
            point = Point(x, y)
            if polygon.contains(point):
                pattern_score = toolbox.unified_detection(k, 0)
                points.append({'lat': y, 'lon': x, 'easting': x, 'northing': y, 'pattern_score': pattern_score})
        if len(points) >= max_points:
            break
    logger.debug(f"Fractal layout: Generated {len(points)} points")
    return points

def generate_chaotic_gcp_layout(polygon, n_gcps, spacing, area_acres):
    """Generate chaotic GCPs with predictive adjustments."""
    toolbox = get_toolbox()
    minx, miny, maxx, maxy = polygon.bounds
    points = []
    attempts = 0
    max_attempts = min(n_gcps * 10, 10000)
    k = int(np.log(area_acres + 1) * 2)
    chaos_factor = toolbox.unified_detection(k, 0, weights=[0.1, 0.1, 0.4, 0.4])
    adjusted_spacing = spacing * (1 + chaos_factor / 10)
    while len(points) < n_gcps and attempts < max_attempts:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        if polygon.contains(Point(x, y)):
            pattern_score = toolbox.unified_detection(len(points), 0)
            points.append({'lat': y, 'lon': x, 'easting': x, 'northing': y, 'pattern_score': pattern_score})
        attempts += 1
    logger.debug(f"Chaotic layout: Generated {attempts} attempts, retained {len(points)} points")
    return points

def generate_gcp(
    geometry_wgs84: Polygon,
    spacing: Union[float, Tuple[float, float]],
    layout_mode: str,
    sequence_type: str,
    modulus: int,
    min_distance_factor: float,
    dem_path: str,
    dsm_path: str,
    fib_sequence: List[int],
    kap_sequence: List[float],
    main_logic: 'MainLogic',
    density_factor: float = 1.0,
    gcp_density: float = 300.0,
    manual_override: bool = False
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Generate GCPs and VCPs for a given layout mode."""
    if isinstance(spacing, (float, int, np.floating)):
        spacing = (float(spacing), float(spacing))
        logger.debug(f"Converted scalar spacing to tuple: {spacing}")

    try:
        utm_crs = get_utm_crs_for_geometry(geometry_wgs84.exterior.coords)
        transformer_to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        transformer_to_wgs84 = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
        geometry_utm = shapely_transform(lambda x, y: transformer_to_utm.transform(x, y), geometry_wgs84)
        if not geometry_utm.is_valid:
            geometry_utm = make_valid(geometry_utm)
        if geometry_utm.is_empty:
            logger.error("Degenerate UTM polygon: area=0")
            return [], [], [], []
        
        minx, miny, maxx, maxy = geometry_utm.bounds
        extent_x, extent_y = maxx - minx, maxy - miny
        area_m2 = geometry_utm.area
        area_acres = calculate_area_acres(area_m2)
        expected_gcp_count = math.ceil(area_m2 / (gcp_density ** 2))
        gcp_points = []
        verification_points = []
        rejected_gcp = []
        rejected_ver = []

        # Handle DEM/DSM paths without context manager
        dem_dataset = None
        dsm_dataset = None
        if dem_path and os.path.exists(dem_path):
            try:
                dem_dataset = rasterio.open(dem_path)
                logger.debug(f"Opened DEM dataset: {dem_path}")
            except Exception as e:
                logger.warning(f"Failed to open DEM dataset {dem_path}: {e}")
                dem_dataset = None
        if dsm_path and os.path.exists(dsm_path):
            try:
                dsm_dataset = rasterio.open(dsm_path)
                logger.debug(f"Opened DSM dataset: {dsm_path}")
            except Exception as e:
                logger.warning(f"Failed to open DSM dataset {dsm_path}: {e}")
                dsm_dataset = None

        try:
            # Helper for layout-specific grid spacings
            def make_grid_spacing(scale_factor=1.0):
                scale = min(0.2, 2.0) / scale_factor
                return (
                    max(extent_x * scale, math.sqrt(area_m2 / expected_gcp_count)),
                    max(extent_y * scale, math.sqrt(area_m2 / expected_gcp_count))
                )

            toolbox = get_toolbox()
            # Generate GCPs
            if layout_mode == "grid":
                grid_spacing = make_grid_spacing(1.0)
                effective_spacing = (gcp_density, gcp_density)  # Force UI-specified density
                x_coords = np.arange(minx, maxx, grid_spacing[0])
                y_coords = np.arange(miny, maxy, grid_spacing[1])
                if len(x_coords) == 0 or len(y_coords) == 0:
                    logger.warning("Adjusted spacing too large for grid; using random points")
                    return [], [], [], []
                initial_count = 0
                for i, y_val in enumerate(y_coords):
                    for j, x_val in enumerate(x_coords):
                        initial_count += 1
                        px = x_val
                        py = y_val
                        point = Point(px, py)
                        if not geometry_utm.contains(point):
                            logger.debug(f"Excluded grid point at ({px:.2f}, {py:.2f}): Outside polygon")
                            continue
                        lon, lat = transformer_to_wgs84.transform(px, py)
                        elevation = get_cached_elevation(lat, lon, dem_path, "DEM", default_elevation=main_logic.centroid_elevation)
                        if elevation is None:
                            logger.debug(f"Excluded grid point at ({px:.2f}, {py:.2f}): Invalid elevation")
                            continue
                        pattern_score = toolbox.unified_detection(len(gcp_points), 0)
                        gcp_points.append({
                            'lat': lat,
                            'lon': lon,
                            'name': f'GCP_{i*len(x_coords)+j+1}',
                            'elevation': elevation,
                            'easting': px,
                            'northing': py,
                            'pattern_score': pattern_score
                        })
                logger.debug(f"Grid layout: Generated {initial_count} candidate GCP points, retained {len(gcp_points)} after filtering")
            elif layout_mode == "triangular":
                grid_spacing = make_grid_spacing(1.0)
                effective_spacing = tuple(s * np.sqrt(8 / np.sqrt(3)) for s in (gcp_density, gcp_density))
                x_coords = np.arange(minx, maxx, grid_spacing[0] / 2)
                y_coords = np.arange(miny, maxy, (grid_spacing[1] * np.sqrt(3)/2) / 2)
                if len(x_coords) == 0 or len(y_coords) == 0:
                    logger.warning("Adjusted spacing too large for triangular grid; using random points")
                    return [], [], [], []
                initial_count = 0
                for i, y_val in enumerate(y_coords):
                    x_offset = (effective_spacing[0] / 2) if i % 2 else 0
                    for j, x_val in enumerate(x_coords):
                        initial_count += 1
                        px = x_val + x_offset
                        py = y_val
                        point = Point(px, py)
                        if not geometry_utm.contains(point):
                            logger.debug(f"Excluded triangular point at ({px:.2f}, {py:.2f}): Outside polygon")
                            continue
                        lon, lat = transformer_to_wgs84.transform(px, py)
                        elevation = get_cached_elevation(lat, lon, dem_path, "DEM", default_elevation=main_logic.centroid_elevation)
                        if elevation is None:
                            logger.debug(f"Excluded triangular point at ({px:.2f}, {py:.2f}): Invalid elevation")
                            continue
                        pattern_score = toolbox.unified_detection(len(gcp_points), 0)
                        gcp_points.append({
                            'lat': lat,
                            'lon': lon,
                            'name': f'GCP_{i*len(x_coords)+j+1}',
                            'elevation': elevation,
                            'easting': px,
                            'northing': py,
                            'pattern_score': pattern_score
                        })
                logger.debug(f"Triangular layout: Generated {initial_count} candidate GCP points, retained {len(gcp_points)} after filtering")
            elif layout_mode == "diamond":
                grid_spacing = make_grid_spacing(1.0)
                effective_spacing = tuple(s * 2 for s in (gcp_density, gcp_density))
                x_coords = np.arange(minx, maxx, grid_spacing[0] / np.sqrt(2))
                y_coords = np.arange(miny, maxy, grid_spacing[1] / np.sqrt(2))
                if len(x_coords) == 0 or len(y_coords) == 0:
                    logger.warning("Adjusted spacing too large for diamond grid; using random points")
                    return [], [], [], []
                initial_count = 0
                for i, y_val in enumerate(y_coords):
                    x_offset = (effective_spacing[0] / np.sqrt(2) / 2) if i % 2 else 0
                    for j, x_val in enumerate(x_coords):
                        initial_count += 1
                        px = x_val + x_offset
                        py = y_val
                        point = Point(px, py)
                        if not geometry_utm.contains(point):
                            logger.debug(f"Excluded diamond point at ({px:.2f}, {py:.2f}): Outside polygon")
                            continue
                        lon, lat = transformer_to_wgs84.transform(px, py)
                        elevation = get_cached_elevation(lat, lon, dem_path, "DEM", default_elevation=main_logic.centroid_elevation)
                        if elevation is None:
                            logger.debug(f"Excluded diamond point at ({px:.2f}, {py:.2f}): Invalid elevation")
                            continue
                        pattern_score = toolbox.unified_detection(len(gcp_points), 0)
                        gcp_points.append({
                            'lat': lat,
                            'lon': lon,
                            'name': f'GCP_{i*len(x_coords)+j+1}',
                            'elevation': elevation,
                            'easting': px,
                            'northing': py,
                            'pattern_score': pattern_score
                        })
                logger.debug(f"Diamond layout: Generated {initial_count} candidate GCP points, retained {len(gcp_points)} after filtering")
            elif layout_mode == "spiral":
                num_points = min(int(area_m2 / (gcp_density ** 2)) * 7, 10000)
                theta = 0.0
                r = 0.0
                spiral_factor = gcp_density / (2 * np.pi)
                attempts = 0
                max_attempts = min(num_points * 50, 10000)
                i = 0
                centroid = geometry_utm.centroid
                logger.debug(f"Attempting to generate {num_points} spiral GCP points, max_attempts={max_attempts}")
                while len(gcp_points) < num_points and attempts < max_attempts:
                    r = np.sqrt(theta) * spiral_factor
                    px = centroid.x + r * np.cos(theta)
                    py = centroid.y + r * np.sin(theta)
                    point = Point(px, py)
                    if geometry_utm.contains(point):
                        lon, lat = transformer_to_wgs84.transform(px, py)
                        elevation = get_cached_elevation(lat, lon, dem_path, "DEM", default_elevation=main_logic.centroid_elevation)
                        if elevation is not None:
                            pattern_score = toolbox.unified_detection(len(gcp_points), 0)
                            gcp_points.append({
                                'lat': lat,
                                'lon': lon,
                                'name': f'GCP_spiral_{i+1}',
                                'elevation': elevation,
                                'easting': px,
                                'northing': py,
                                'pattern_score': pattern_score
                            })
                            i += 1
                        else:
                            logger.debug(f"Excluded spiral point at ({px:.2f}, {py:.2f}): Invalid elevation")
                    else:
                        logger.debug(f"Excluded spiral point at ({px:.2f}, {py:.2f}): Outside polygon")
                    theta += 0.1
                    attempts += 1
                logger.debug(f"Spiral layout: Generated {attempts} attempts, retained {len(gcp_points)} GCP points")
            elif layout_mode == "fractal":
                gcp_points = generate_fractal_gcp_layout(geometry_utm, gcp_density, area_acres, expected_gcp_count)
                for i, pt in enumerate(gcp_points):
                    lon, lat = transformer_to_wgs84.transform(pt['easting'], pt['northing'])
                    elevation = get_cached_elevation(lat, lon, dem_path, "DEM", default_elevation=main_logic.centroid_elevation)
                    pt.update({
                        'lat': lat,
                        'lon': lon,
                        'name': f'GCP_fractal_{i+1}',
                        'elevation': elevation if elevation is not None else main_logic.centroid_elevation
                    })
            elif layout_mode == "chaotic":
                n_gcps = min(int(area_m2 / (gcp_density ** 2)) * 7, 10000)
                gcp_points = generate_chaotic_gcp_layout(geometry_utm, n_gcps, gcp_density, area_acres)
                for i, pt in enumerate(gcp_points):
                    lon, lat = transformer_to_wgs84.transform(pt['easting'], pt['northing'])
                    elevation = get_cached_elevation(lat, lon, dem_path, "DEM", default_elevation=main_logic.centroid_elevation)
                    pt.update({
                        'lat': lat,
                        'lon': lon,
                        'name': f'GCP_chaotic_{i+1}',
                        'elevation': elevation if elevation is not None else main_logic.centroid_elevation
                    })

            # Filter GCPs for minimum distance within GCP network
            if gcp_points:
                kept_gcp, rej_gcp, _ = reject_coords_too_close(
                    gcp_points, area_acres, gcp_density, manual_override, point_type="GCP"
                )
                gcp_points = kept_gcp
                rejected_gcp.extend(rej_gcp)
                logger.debug(f"After GCP deduplication: {len(gcp_points)} GCPs, {len(rejected_gcp)} rejected")

            # Generate VCPs (10% more than GCPs)
            target_vcp_count = math.ceil(len(gcp_points) * 1.1)
            vcp_spacing = tuple(s * 1.1 for s in (gcp_density, gcp_density))
            if layout_mode == "grid":
                grid_spacing = make_grid_spacing(1.1)
                x_coords = np.arange(minx, maxx, grid_spacing[0])
                y_coords = np.arange(miny, maxy, grid_spacing[1])
                if len(x_coords) == 0 or len(y_coords) == 0:
                    logger.warning("Adjusted spacing too large for VCP grid; using random points")
                    return gcp_points, [], rejected_gcp, []
                initial_count = 0
                for i, y_val in enumerate(y_coords):
                    for j, x_val in enumerate(x_coords):
                        initial_count += 1
                        px = x_val + grid_spacing[0] * 0.5  # Offset VCPs
                        py = y_val + grid_spacing[1] * 0.5
                        point = Point(px, py)
                        if not geometry_utm.contains(point):
                            logger.debug(f"Excluded VCP grid point at ({px:.2f}, {py:.2f}): Outside polygon")
                            continue
                        lon, lat = transformer_to_wgs84.transform(px, py)
                        elevation = get_cached_elevation(lat, lon, dem_path, "DEM", default_elevation=main_logic.centroid_elevation)
                        if elevation is None:
                            logger.debug(f"Excluded VCP grid point at ({px:.2f}, {py:.2f}): Invalid elevation")
                            continue
                        pattern_score = toolbox.unified_detection(len(verification_points), 0)
                        verification_points.append({
                            'lat': lat,
                            'lon': lon,
                            'name': f'VCP_{i*len(x_coords)+j+1}',
                            'elevation': elevation,
                            'easting': px,
                            'northing': py,
                            'pattern_score': pattern_score
                        })
                        if len(verification_points) >= target_vcp_count:
                            break
                    if len(verification_points) >= target_vcp_count:
                        break
                logger.debug(f"VCP Grid layout: Generated {initial_count} candidate VCP points, retained {len(verification_points)}")
            elif layout_mode == "spiral":
                num_points = min(target_vcp_count, 10000)
                theta = 0.0
                r = 0.0
                spiral_factor = vcp_spacing[0] / (2 * np.pi)
                attempts = 0
                max_attempts = min(num_points * 50, 10000)
                i = 0
                centroid = geometry_utm.centroid
                logger.debug(f"Attempting to generate {num_points} spiral VCP points, max_attempts={max_attempts}")
                while len(verification_points) < num_points and attempts < max_attempts:
                    r = np.sqrt(theta) * spiral_factor
                    px = centroid.x + r * np.cos(theta + np.pi)
                    py = centroid.y + r * np.sin(theta + np.pi)
                    point = Point(px, py)
                    if geometry_utm.contains(point):
                        lon, lat = transformer_to_wgs84.transform(px, py)
                        elevation = get_cached_elevation(lat, lon, dem_path, "DEM", default_elevation=main_logic.centroid_elevation)
                        if elevation is not None:
                            pattern_score = toolbox.unified_detection(len(verification_points), 0)
                            verification_points.append({
                                'lat': lat,
                                'lon': lon,
                                'name': f'VCP_spiral_{i+1}',
                                'elevation': elevation,
                                'easting': px,
                                'northing': py,
                                'pattern_score': pattern_score
                            })
                            i += 1
                        else:
                            logger.debug(f"Excluded VCP spiral point at ({px:.2f}, {py:.2f}): Invalid elevation")
                    else:
                        logger.debug(f"Excluded VCP spiral point at ({px:.2f}, {py:.2f}): Outside polygon")
                    theta += 0.1
                    attempts += 1
                logger.debug(f"VCP Spiral layout: Generated {attempts} attempts, retained {len(verification_points)} VCP points")
            else:
                attempts = 0
                max_attempts = min(target_vcp_count * 50, 10000)
                while len(verification_points) < target_vcp_count and attempts < max_attempts:
                    px = random.uniform(minx, maxx)
                    py = random.uniform(miny, maxy)
                    point = Point(px, py)
                    if geometry_utm.contains(point):
                        lon, lat = transformer_to_wgs84.transform(px, py)
                        elevation = get_cached_elevation(lat, lon, dem_path, "DEM", default_elevation=main_logic.centroid_elevation)
                        if elevation is not None:
                            pattern_score = toolbox.unified_detection(len(verification_points), 0)
                            verification_points.append({
                                'lat': lat,
                                'lon': lon,
                                'name': f'VCP_chaotic_{len(verification_points)+1}',
                                'elevation': elevation,
                                'easting': px,
                                'northing': py,
                                'pattern_score': pattern_score
                            })
                        else:
                            logger.debug(f"Excluded VCP chaotic point at ({px:.2f}, {py:.2f}): Invalid elevation")
                    attempts += 1
                logger.debug(f"VCP Chaotic layout: Generated {attempts} attempts, retained {len(verification_points)} VCP points")

            # Filter VCPs for minimum distance within VCP network
            if verification_points:
                kept_vcp, rej_vcp, _ = reject_coords_too_close(
                    verification_points, area_acres, gcp_density, manual_override, point_type="VCP"
                )
                verification_points = kept_vcp
                rejected_ver.extend(rej_vcp)
                logger.debug(f"After VCP deduplication: {len(verification_points)} VCPs, {len(rejected_ver)} rejected")

            # Elevations for all points
            latlon_points = [(p['lat'], p['lon']) for p in gcp_points + verification_points]
            if latlon_points:
                try:
                    elevations, _ = get_srtm_elevation_bulk(latlon_points, retry=3)
                    for i, (pt, elev) in enumerate(zip(gcp_points + verification_points, elevations)):
                        if elev is not None and elev > 0:
                            pt['elevation'] = float(elev)
                        else:
                            pt['elevation'] = main_logic.centroid_elevation
                except Exception as e:
                    logger.warning(f"Elevation bulk query failed: {e}")
                    for pt in gcp_points + verification_points:
                        pt['elevation'] = main_logic.centroid_elevation

            logger.info(f"Generated {len(gcp_points)} GCPs, {len(verification_points)} VCPs")
            return gcp_points, verification_points, rejected_gcp, rejected_ver
        finally:
            if dem_dataset:
                dem_dataset.close()
            if dsm_dataset:
                dsm_dataset.close()
    except Exception as e:
        logger.error(f"GCP generation failed: {str(e)}")
        return [], [], [], []