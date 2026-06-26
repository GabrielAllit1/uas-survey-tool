import numpy as np
from shapely.geometry import Point, Polygon
import rasterio
from unified_mathtoolbox import MathToolBox
from elevation_service import detect_elevation_cycles

def load_dsm(filepath):
    """Load DSM file using rasterio."""
    try:
        with rasterio.open(filepath) as dataset:
            return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load DSM file: {str(e)}")

def sample_dsm(dataset, lat, lon):
    """Sample DSM elevation at a lat/lon point."""
    try:
        row, col = dataset.index(lon, lat)
        if 0 <= row < dataset.height and 0 <= col < dataset.width:
            value = dataset.read(1)[row, col]
            return value if value != dataset.nodata else None
        return None
    except Exception:
        return None

def filter_points_with_dsm(polygon_coords, spacing, dsm=None):
    """
    Generate candidate points and filter based on DSM data with relaxed thresholds.
    Returns (candidates, removed, kept, cycle_score, pattern_score).
    """
    if dsm is None:
        raise ValueError("DSM path must be provided.")

    dataset = load_dsm(dsm)
    poly = Polygon(polygon_coords)

    minx, miny, maxx, maxy = poly.bounds
    step = spacing / 111320  # Convert spacing (m) to degrees

    # Generate denser grid with cap on total points
    max_points = 10000  # Prevent excessive point generation
    x_vals = np.arange(minx, maxx, step * 0.8)[:int(np.sqrt(max_points))]
    y_vals = np.arange(miny, maxy, step * 0.8)[:int(np.sqrt(max_points))]
    
    candidates = []
    removed = []
    kept = []
    elevations = []

    toolbox = MathToolBox()
    for x in x_vals:
        for y in y_vals:
            if len(candidates) >= max_points:
                break
            point = Point(x, y)
            if poly.contains(point):
                elevation = sample_dsm(dataset, y, x)
                if elevation is not None and elevation < 9999:
                    candidates.append((x, y))
                    elevations.append(elevation)
                    pattern_score = toolbox.unified_detection(len(candidates), 0)
                    # Relaxed elevation threshold
                    elevation_threshold = 50 * (1 + pattern_score)  # Increased from 25
                    if elevation > elevation_threshold:
                        removed.append((x, y))
                    else:
                        kept.append((x, y))
        if len(candidates) >= max_points:
            break

    cycle_score = detect_elevation_cycles(elevations) if elevations else 0.0
    pattern_score = toolbox.unified_detection(len(candidates), 0) if candidates else 0.0
    print(f"[DEBUG] Vegetation Filter: Cycle Score: {cycle_score:.2f}, Pattern Score: {pattern_score:.2f}")

    return candidates, removed, kept, cycle_score, pattern_score