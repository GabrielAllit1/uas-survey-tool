import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree
import logging
import os
from unified_mathtoolbox import MathToolBox

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.expanduser("~/UAS_Survey_Tool_Logs"), 'filter_coordinates.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Singleton-like MathToolBox instance
_toolbox = None
def get_toolbox():
    global _toolbox
    if _toolbox is None:
        _toolbox = MathToolBox()
        logger.debug("Initialized singleton MathToolBox")
    return _toolbox

def fractal_score(point, area_acres):
    """Calculate fractal score for a point based on area and Fibonacci sequence."""
    toolbox = get_toolbox()
    fib = [toolbox.fibonacci(i) for i in range(21)]
    k = int(np.log(area_acres + 1) * 2) % len(fib)
    index = (2 * fib[k] - 2) % len(fib)
    return fib[index] / fib[8] if fib[8] != 0 else 1.0

def apply_obstruction_filter(gcp_points, obstruction_mask=None, buffer_radius=2.0, area_acres=1.0):
    """
    Filter GCP points based on obstruction mask.
    """
    if not obstruction_mask or not isinstance(obstruction_mask, list):
        toolbox = get_toolbox()
        pattern_score = toolbox.unified_detection(len(gcp_points), 0)
        logger.debug(f"No obstruction mask provided; retaining {len(gcp_points)} points with pattern_score={pattern_score:.2f}")
        return gcp_points, pattern_score

    tree = STRtree(obstruction_mask)
    filtered_gcps = []
    max_points = 10000

    for gcp in gcp_points[:max_points]:
        pt = Point(gcp['easting'], gcp['northing']).buffer(buffer_radius)
        nearby = tree.query(pt)
        if not nearby:
            filtered_gcps.append(gcp)
    
    toolbox = get_toolbox()
    pattern_score = toolbox.unified_detection(len(filtered_gcps), 0)
    logger.debug(f"Applied obstruction filter: {len(filtered_gcps)} points retained out of {min(len(gcp_points), max_points)}")
    return filtered_gcps, pattern_score

def filter_coordinates_by_elevation(points, elevations, threshold=45.0, area_acres=1.0, dem_path=None):
    """
    Filter points based on elevation variance and terrain analysis.
    """
    if len(points) != len(elevations):
        logger.warning(f"Mismatch between points ({len(points)}) and elevations ({len(elevations)})")
        return points, 0.0

    max_points = 10000
    filtered_points = []
    elevation_diffs = []
    bins = 10
    bin_counts = np.zeros(bins)

    for i, (point, elev) in enumerate(zip(points[:max_points], elevations[:max_points])):
        if elev is None:
            logger.debug(f"Skipping point at ({point['easting']:.2f}, {point['northing']:.2f}): Invalid elevation")
            continue
        elevation_diffs.append(elev - np.mean([e for e in elevations[:max_points] if e is not None]))
    
    if not elevation_diffs:
        logger.debug("No valid elevation differences; retaining all points")
        toolbox = get_toolbox()
        return points, toolbox.unified_detection(len(points), 0)

    abs_diffs = np.abs(elevation_diffs)
    max_diff = max(abs_diffs) if abs_diffs else threshold
    bin_edges = np.linspace(0, max_diff, bins + 1)

    for i, (point, diff) in enumerate(zip(points[:max_points], abs_diffs)):
        if diff <= threshold:
            bin_idx = np.digitize(diff, bin_edges) - 1
            bin_idx = min(bin_idx, bins - 1)
            toolbox = get_toolbox()
            pattern_score = toolbox.unified_detection(len(filtered_points), 0)
            point['pattern_score'] = pattern_score
            filtered_points.append(point)
            bin_counts[bin_idx] += 1

    toolbox = get_toolbox()
    pattern_score = toolbox.unified_detection(len(filtered_points), 0)
    logger.debug(f"Applied elevation filter: {len(filtered_points)} points retained out of {min(len(points), max_points)}")
    return filtered_points, pattern_score

def reject_coords_too_close(points, area_acres=1.0, gcp_density=400.0, manual_override=False, point_type="GCP"):
    """
    Filter points that are too close within their own network (GCP or VCP) based on UI-specified GCP density.

    Args:
        points: List of point dictionaries with 'easting', 'northing'.
        area_acres: Survey area in acres for scoring.
        gcp_density: UI-specified GCP spacing density in meters (e.g., 400m).
        manual_override: If True, bypass min_distance restrictions.
        point_type: "GCP" or "VCP" to identify the network.

    Returns:
        Tuple: (kept points, rejected points, pattern score).
    """
    kept = []
    rejected = []
    max_points = 10000
    min_distance = gcp_density * 0.25 if not manual_override else 0.0  # 25% of GCP density unless overridden

    for pt in points[:max_points]:
        score = fractal_score(pt, area_acres)
        adjusted_distance = min_distance / (score * 10) if not manual_override else 0.0
        new_point = Point(pt['easting'], pt['northing'])
        too_close = False
        for existing in kept:
            if new_point.distance(Point(existing['easting'], existing['northing'])) < adjusted_distance:
                logger.debug(f"Excluded {point_type} point at ({pt['easting']:.2f}, {pt['northing']:.2f}): Too close to existing {point_type} point (distance < {adjusted_distance:.2f}m)")
                rejected.append(pt)
                too_close = True
                break
        if not too_close:
            toolbox = get_toolbox()
            pattern_score = toolbox.unified_detection(len(kept), 0)
            pt['pattern_score'] = pattern_score
            kept.append(pt)

    toolbox = get_toolbox()
    pattern_score = toolbox.unified_detection(len(kept), 0)
    logger.debug(f"Rejected close {point_type} coordinates: {len(kept)} points retained, {len(rejected)} points rejected out of {min(len(points), max_points)} with min_distance={min_distance:.2f}m")
    return kept, rejected, pattern_score