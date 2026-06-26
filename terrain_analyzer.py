import numpy as np
from shapely.geometry import Point
import rasterio
from unified_mathtoolbox import MathToolBox
from elevation_service import detect_elevation_cycles

def load_terrain_elevation(tif_path):
    """Load terrain elevation data from a TIFF file."""
    try:
        with rasterio.open(tif_path) as dataset:
            elevation = dataset.read(1)
            transform = dataset.transform
            return elevation, transform, dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load terrain file: {str(e)}")

def get_elevation_at_coords(coords, elevation, transform, dataset):
    """
    Get elevation at given coordinates.
    Args:
        coords: (lon, lat) tuple
        elevation: Elevation array from rasterio
        transform: Rasterio Affine transform
        dataset: Rasterio dataset for bounds checking
    Returns:
        Elevation value or None if out of bounds
    """
    lon, lat = coords
    col, row = ~transform * (lon, lat)
    row, col = int(row), int(col)
    try:
        if 0 <= row < dataset.height and 0 <= col < dataset.width:
            return elevation[row, col]
        return None
    except IndexError:
        return None

def filter_gcps_by_slope_and_height(gcps, elevation, transform, dataset, max_slope=15, max_variance=20):
    """
    Filter GCPs based on slope and elevation variance with adaptive thresholds.
    Returns refined GCPs and pattern score.
    """
    refined = []
    elevations = []
    max_points = 10000  # Cap to prevent excessive processing
    for gcp in gcps[:max_points]:
        elev = get_elevation_at_coords((gcp['lon'], gcp['lat']), elevation, transform, dataset)
        if elev is not None:
            gcp['elevation'] = elev
            elevations.append(elev)
            refined.append(gcp)
    
    if not elevations:
        return refined, 0.0
    
    mean = np.mean(elevations)
    cycle_score = detect_elevation_cycles(elevations)
    toolbox = MathToolBox()
    pattern_score = toolbox.unified_detection(len(refined), 0)
    # Adaptive variance based on area complexity
    adjusted_variance = max_variance * (1 + cycle_score * 1.5)  # Increased flexibility
    refined = [g for g in refined if abs(g['elevation'] - mean) < adjusted_variance]
    
    print(f"[DEBUG] Terrain Filter: Cycle Score: {cycle_score:.2f}, Pattern Score: {pattern_score:.2f}")
    return refined, pattern_score