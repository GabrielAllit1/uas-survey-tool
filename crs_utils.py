from __future__ import annotations
from typing import Iterable, Tuple, Optional, Union
from pyproj import CRS

LonLat = Tuple[float, float]

def deduce_project_crs(
    polygon_coords: Optional[Iterable[LonLat]],
    fallback: Union[str, CRS] = "EPSG:4326"
) -> CRS:
    """
    Pick a sensible projected CRS based on the AOI centroid.
    - Uses UTM zone derived from centroid lat/lon.
    - Works for any hemisphere.
    - Falls back to WGS84 if polygon is missing.

    Returns a pyproj.CRS object.
    """
    try:
        if polygon_coords:
            lons = [float(x) for x, _ in polygon_coords]
            lats = [float(y) for _, y in polygon_coords]
            clon = sum(lons) / len(lons)
            clat = sum(lats) / len(lats)
            zone = int((clon + 180) // 6) + 1
            epsg = 32600 + zone if clat >= 0 else 32700 + zone
            return CRS.from_epsg(epsg)
    except Exception:
        pass
    return CRS.from_user_input(fallback)
