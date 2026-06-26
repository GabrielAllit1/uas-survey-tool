import math
import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def calculate_flight_parameters(
    area_m2: float,
    altitude_m: float,
    sensor_width_mm: float,
    focal_length_mm: float,
    image_width_px: int,
    speed_mps: float,
    swath_width_m: float,
    layout_mode: str = 'grid',
    desired_rmse_m: float = 0.06096,
    terrain_elevation_variance: float = 10.0
) -> Optional[Dict[str, float]]:
    """
    Calculate flight parameters for UAS survey to achieve desired RMSE.

    Args:
        area_m2: Survey area in square meters.
        altitude_m: Flight altitude in meters.
        sensor_width_mm: Camera sensor width in millimeters.
        focal_length_mm: Camera focal length in millimeters.
        image_width_px: Image width in pixels.
        speed_mps: UAV speed in meters per second.
        swath_width_m: Swath width in meters.
        layout_mode: Layout mode ('grid', 'spiral', etc.).
        desired_rmse_m: Desired RMSE in meters (default 0.06096m for 0.2 US Survey feet).
        terrain_elevation_variance: Variance of terrain elevation for density adjustment.

    Returns:
        Optional[Dict[str, float]]: Dictionary of flight parameters, or None if calculation fails.
    """
    try:
        # Validate inputs
        if any(param <= 0 for param in [area_m2, altitude_m, sensor_width_mm, focal_length_mm, image_width_px, speed_mps, swath_width_m]):
            logger.error("Invalid flight parameter values: %s",
                         {"area_m2": area_m2, "altitude_m": altitude_m, "sensor_width_mm": sensor_width_mm,
                          "focal_length_mm": focal_length_mm, "image_width_px": image_width_px,
                          "speed_mps": speed_mps, "swath_width_m": swath_width_m})
            return None

        # Calculate ground sample distance (GSD)
        gsd_m = (sensor_width_mm / 1000) * (altitude_m / focal_length_mm) / image_width_px
        gsd_mm = gsd_m * 1000  # Convert to millimeters

        # Base spacing constant based on layout mode
        k = {
            'grid': 50.0,
            'triangular': 45.0,
            'diamond': 48.0,
            'spiral': 40.0,
            'fractal': 35.0,
            'chaotic': 30.0
        }.get(layout_mode, 50.0)

        # Adjust spacing for terrain complexity
        terrain_factor = 1.0 + (terrain_elevation_variance / 50.0 if terrain_elevation_variance > 0 else 0.2)
        recommended_spacing_m = min(100.0, max(10.0, math.sqrt(area_m2) / (k * desired_rmse_m * terrain_factor)))

        # Estimate GCP count based on area, spacing, and RMSE
        accuracy_factor = 1.0 / (desired_rmse_m / 0.06096)  # Normalize to target RMSE
        estimated_gcp_count = max(10, int(area_m2 / (recommended_spacing_m ** 2) * accuracy_factor * terrain_factor))
        estimated_gcp_count = min(150, estimated_gcp_count)  # Cap at reasonable limit

        # Estimate VCP count (25% more than GCP count)
        estimated_vcp_count = int(estimated_gcp_count * 1.25)

        # Calculate Points Per Square Meter (PPSM)
        ppsm = estimated_gcp_count / area_m2 if area_m2 > 0 else 0.0

        # Pattern score based on layout mode and RMSE achievement
        pattern_score = min(1.0, max(0.0, 1.0 - (desired_rmse_m / gsd_m))) if gsd_m > 0 else 0.6
        pattern_score *= {'grid': 1.0, 'triangular': 1.0, 'diamond': 0.9, 'spiral': 0.8, 'fractal': 0.7, 'chaotic': 0.6}.get(layout_mode, 0.6)

        # Calculate coverage time
        coverage_time_s = area_m2 / (swath_width_m * speed_mps) if swath_width_m > 0 and speed_mps > 0 else 0.0

        params = {
            'gsd_mm': gsd_mm,
            'recommended_spacing_m': recommended_spacing_m,
            'estimated_gcp_count': estimated_gcp_count,
            'estimated_vcp_count': estimated_vcp_count,
            'ppsm': ppsm,
            'pattern_score': pattern_score,
            'coverage_time_s': coverage_time_s,
            'altitude_m': altitude_m,
            'sensor_width_mm': sensor_width_mm,
            'focal_length_mm': focal_length_mm,
            'image_width_px': image_width_px,
            'speed_mps': speed_mps,
            'swath_width_m': swath_width_m
        }

        logger.debug("Computed flight parameters: %s", params)
        return params
    except Exception as e:
        logger.error("Flight parameter calculation failed: %s", str(e))
        return None