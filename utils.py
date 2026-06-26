import logging
import os
from pyproj import CRS

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.expanduser("~/UAS_Survey_Tool_Logs"), 'uas_survey_tool.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def get_utm_crs_for_geometry(coords):
    """
    Determine the UTM zone from a list of coordinates.

    Args:
        coords (list): List of (lat, lon) tuples.

    Returns:
        CRS: The UTM CRS object for the zone.
    """
    try:
        if not coords:
            raise ValueError("No coordinates provided")

        # Calculate average latitude and longitude
        avg_lat = sum(lat for lat, lon in coords) / len(coords)
        avg_lon = sum(lon for lat, lon in coords) / len(coords)
        zone = int((avg_lon + 180) / 6) + 1
        hemisphere = 'north' if avg_lat >= 0 else 'south'

        # Create and return CRS object
        utm_crs = CRS.from_dict({'proj': 'utm', 'zone': zone, 'south': hemisphere == 'south'})
        logging.debug(f"Calculated UTM CRS: {utm_crs} for avg_lat={avg_lat:.2f}, avg_lon={avg_lon:.2f}")
        return utm_crs

    except Exception as e:
        logging.error(f"UTM zone calculation failed: {e}")
        return CRS.from_epsg(4326)  # Fallback to WGS84