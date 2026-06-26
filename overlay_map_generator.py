# overlay_map_generator.py
from __future__ import annotations
import os
import logging
from typing import Iterable, Dict, Any, Optional

# Must set backend before importing pyplot (we render to file only)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import transform as shapely_transform
from pyproj import CRS, Transformer

_LOGGER = logging.getLogger(__name__)
if not _LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def auto_detect_utm_crs(lon: float, lat: float, return_crs_object: bool = True):
    """
    Returns an appropriate UTM CRS for a lon/lat.
    If return_crs_object is False, returns an 'EPSG:xxxx' string.
    """
    try:
        if lon is None or lat is None:
            return CRS.from_epsg(4326) if return_crs_object else "EPSG:4326"
        zone = int((lon + 180) // 6) + 1
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        return CRS.from_epsg(epsg) if return_crs_object else f"EPSG:{epsg}"
    except Exception as exc:
        _LOGGER.warning("auto_detect_utm_crs failed for (%.6f, %.6f): %s", lon, lat, exc)
        return CRS.from_epsg(4326) if return_crs_object else "EPSG:4326"


def _to_wgs84(geom, src_crs) -> Polygon | MultiPolygon:
    """
    Safely project *geom* from src_crs -> EPSG:4326.
    If src_crs is already WGS84 or None, returns geom as-is.
    """
    try:
        if src_crs is None:
            return geom
        src = CRS.from_user_input(src_crs)
        if src.to_epsg() == 4326:
            return geom
        tfm = Transformer.from_crs(src, "EPSG:4326", always_xy=True)
        return shapely_transform(lambda x, y: tfm.transform(x, y), geom)
    except Exception as exc:
        _LOGGER.warning("Projection to WGS84 failed: %s. Returning original geometry.", exc)
        return geom


def _prep_xy_lists(geom: Polygon | MultiPolygon):
    xs, ys = [], []
    def _append_ring(ring):
        x, y = ring.xy
        xs.append(list(x))
        ys.append(list(y))

    if isinstance(geom, Polygon):
        _append_ring(geom.exterior)
        for interior in geom.interiors:
            _append_ring(interior)
    elif isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            _append_ring(g.exterior)
            for interior in g.interiors:
                _append_ring(interior)
    else:
        raise TypeError(f"Unsupported geometry type: {type(geom)}")
    return xs, ys


def _collect_xy_from_points(pts: Iterable[Dict[str, Any]]) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for p in pts or []:
        # Prefer lon/lat if provided; fall back to easting/northing if they look like lon/lat
        lon = p.get("lon")
        lat = p.get("lat")
        if lon is not None and lat is not None:
            xs.append(float(lon))
            ys.append(float(lat))
            continue
        # Fallback: try easting/northing, but only if they already seem near lon/lat range
        e = p.get("easting")
        n = p.get("northing")
        if e is not None and n is not None and -180.0 <= float(e) <= 180.0 and -90.0 <= float(n) <= 90.0:
            xs.append(float(e))
            ys.append(float(n))
    return xs, ys


def generate_overlay_map(
    output_path: str,
    geometry_wgs84,   # shapely Polygon/MultiPolygon in WGS84 (or anything + crs arg, we'll reproject)
    gcp_points: Optional[list[Dict[str, Any]]] = None,
    verification_points: Optional[list[Dict[str, Any]]] = None,
    rejected_gcp_points: Optional[list[Dict[str, Any]]] = None,
    rejected_verification_points: Optional[list[Dict[str, Any]]] = None,
    crs: Optional[Any] = None,
    dpi: int = 220,
) -> str:
    """
    Simple static overlay figure (PNG). Assumes *geometry_wgs84* is in EPSG:4326,
    but if *crs* is provided and not EPSG:4326, we will project to WGS84.
    """
    if geometry_wgs84 is None:
        raise ValueError("generate_overlay_map: geometry_wgs84 is required")

    # Guarantee AOI is in lon/lat for plotting
    geom_ll = _to_wgs84(geometry_wgs84, crs)

    # 1) Setup plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # 2) Draw AOI boundary
    xs_list, ys_list = _prep_xy_lists(geom_ll)
    for xs, ys in zip(xs_list, ys_list):
        ax.plot(xs, ys, linewidth=2.0, alpha=0.95, label="AOI" if ax.get_legend() is None else None)

    # 3) Draw points
    def _scatter(pts, marker, label):
        xs, ys = _collect_xy_from_points(pts)
        if xs and ys:
            ax.scatter(xs, ys, marker=marker, s=22, alpha=0.9, label=label)

    _scatter(gcp_points, "o", "GCP")
    _scatter(verification_points, "^", "VCP")
    _scatter(rejected_gcp_points, "x", "Rejected GCP")
    _scatter(rejected_verification_points, "x", "Rejected VCP")

    # 4) Extent & cosmetics
    minx, miny, maxx, maxy = geom_ll.bounds
    pad_x = (maxx - minx) * 0.05 or 0.001
    pad_y = (maxy - miny) * 0.05 or 0.001
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title("AOI with GCP/VCP Overlay", fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    # 5) Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    _LOGGER.info("Overlay map saved: %s", output_path)
    return output_path
