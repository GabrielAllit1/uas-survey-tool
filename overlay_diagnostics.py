# overlay_diagnostics.py
from __future__ import annotations

import os
import tempfile
import logging
from typing import Iterable, Tuple, Optional, Any, Dict, List, Union

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, MultiPolygon, Point as ShpPoint
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform

try:
    from pyproj import CRS, Transformer
except Exception:
    CRS = None
    Transformer = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def _to_utm(geom: BaseGeometry, crs_like: Any) -> BaseGeometry:
    """Project any geometry to a given CRS (string or pyproj.CRS). If not available, return as-is."""
    if Transformer is None:
        return geom
    try:
        if crs_like is None:
            return geom
        src = CRS.from_epsg(4326)
        dst = CRS.from_user_input(crs_like)
        if src == dst:
            return geom
        tf = Transformer.from_crs(src, dst, always_xy=True)
        return shapely_transform(lambda x, y: tf.transform(x, y), geom)
    except Exception as e:
        logger.warning("Projection to UTM failed: %s. Using original geometry.", e)
        return geom


def _to_wgs84(geom: BaseGeometry, crs_like: Any) -> BaseGeometry:
    """Project any geometry to WGS84 if a CRS is provided, else return as-is."""
    if Transformer is None:
        return geom
    try:
        if crs_like is None:
            return geom
        src = CRS.from_user_input(crs_like)
        if src.to_epsg() == 4326:
            return geom
        tf = Transformer.from_crs(src, "EPSG:4326", always_xy=True)
        return shapely_transform(lambda x, y: tf.transform(x, y), geom)
    except Exception as e:
        logger.warning("Projection to WGS84 failed: %s. Using original geometry.", e)
        return geom


def _build_polygon_from_coords(coords: Iterable[Tuple[float, float]]) -> Optional[Polygon]:
    pts = [(float(lon), float(lat)) for lon, lat in coords or []]
    if len(pts) < 3:
        return None
    try:
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if poly.is_valid else None
    except Exception:
        return None


def _collect_xy_from_point_dicts(
    pts: Optional[List[Dict[str, Any]]],
    want_utm: bool,
    utm_crs: Optional[Any]
) -> Tuple[List[float], List[float]]:
    xs, ys = [], []
    if not pts:
        return xs, ys

    tf_to_utm = None
    tf_to_ll = None
    if Transformer is not None and utm_crs is not None:
        try:
            tf_to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
            tf_to_ll = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
        except Exception:
            tf_to_utm = None
            tf_to_ll = None

    for p in pts:
        # Prefer explicit easting/northing if we want UTM and they exist
        e = p.get("easting")
        n = p.get("northing")
        lon = p.get("lon")
        lat = p.get("lat")

        if want_utm:
            if e is not None and n is not None:
                xs.append(float(e)); ys.append(float(n)); continue
            # Try transform lon/lat -> UTM
            if lon is not None and lat is not None and tf_to_utm is not None:
                ex, ny = tf_to_utm.transform(float(lon), float(lat))
                xs.append(float(ex)); ys.append(float(ny)); continue
            # Last resort: if lon/lat look like lon/lat but no transformer, plot them as-is
            if lon is not None and lat is not None:
                xs.append(float(lon)); ys.append(float(lat)); continue
        else:
            # want lon/lat
            if lon is not None and lat is not None:
                xs.append(float(lon)); ys.append(float(lat)); continue
            if e is not None and n is not None and tf_to_ll is not None:
                lo, la = tf_to_ll.transform(float(e), float(n))
                xs.append(float(lo)); ys.append(float(la)); continue
            if e is not None and n is not None:
                xs.append(float(e)); ys.append(float(n)); continue
    return xs, ys


def _plot_polygon(ax, poly: BaseGeometry, label: str = "Survey Area"):
    if not isinstance(poly, BaseGeometry) or poly.is_empty:
        return
    if isinstance(poly, (Polygon, )):
        x, y = poly.exterior.xy
        ax.plot(x, y, "k-", linewidth=1.2, alpha=0.9, label=label)
        for ring in poly.interiors:
            xi, yi = ring.xy
            ax.plot(xi, yi, "k--", linewidth=0.6, alpha=0.6)
    elif isinstance(poly, MultiPolygon):
        for g in poly.geoms:
            _plot_polygon(ax, g, label)
    else:
        try:
            x, y = poly.exterior.xy
            ax.plot(x, y, "k-", linewidth=1.2, alpha=0.9, label=label)
        except Exception:
            pass


def generate_overlay_diagnostics(
    # original/strict signature bits
    polygon: Optional[BaseGeometry] = None,
    utm_crs: Optional[Any] = None,
    gcp_points: Optional[Iterable[Tuple[float, float]]] = None,
    veri_points: Optional[Iterable[Tuple[float, float]]] = None,
    dem_path: Optional[str] = None,
    dsm_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    output_path: Optional[str] = None,
    # flexible kwargs your UI sometimes sends
    polygon_coords: Optional[Iterable[Tuple[float, float]]] = None,
    elevation_data: Optional[Dict[str, Any]] = None,
    detection_score: Optional[float] = None,
    **kwargs
) -> dict:
    """
    Creates an overlay PNG with AOI boundary and points.

    Accepts either:
      • a shapely *polygon* (preferably in UTM) OR
      • *polygon_coords* (lon,lat) list that we will build into a polygon.

    Points can be:
      • sequences of (x, y) already in UTM, OR
      • dicts with 'easting'/'northing', OR
      • dicts with 'lon'/'lat' (we'll transform to UTM if possible).

    Returns:
      {
        "image_path": <str or None>,
        "gcp_count": <int>,
        "vcp_count": <int>,
        "summary": <str>
      }
    """
    try:
        # Resolve output path
        if not output_path:
            if not output_dir:
                output_dir = tempfile.mkdtemp(prefix="uas_overlay_")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "overlay.png")

        # Build polygon from coords if needed
        if polygon is None and polygon_coords:
            # polygon_coords expected in lon/lat (EPSG:4326)
            poly_ll = _build_polygon_from_coords(polygon_coords)
            if poly_ll is not None and utm_crs is not None:
                polygon = _to_utm(poly_ll, utm_crs)
            else:
                polygon = poly_ll

        # Determine plotting coordinate space
        # If we have a UTM CRS, we try to plot in UTM; otherwise, we plot lon/lat.
        want_utm = utm_crs is not None

        # Normalize points to plotting space (either UTM or lon/lat)
        def _normalize_points(pts: Optional[Iterable]) -> List[Tuple[float, float]]:
            if not pts:
                return []
            pts_list: List[Tuple[float, float]] = []
            if isinstance(pts, list) and pts and isinstance(pts[0], dict):
                xs, ys = _collect_xy_from_point_dicts(pts, want_utm=want_utm, utm_crs=utm_crs)
                pts_list = list(zip(xs, ys))
            else:
                # assume (x, y) tuples already in the right frame
                try:
                    pts_list = [(float(x), float(y)) for (x, y) in pts]
                except Exception:
                    pts_list = []
            return pts_list

        gcp_xy = _normalize_points(gcp_points)
        vcp_xy = _normalize_points(veri_points)

        # Plot
        fig, ax = plt.subplots(figsize=(9.5, 7.0))

        if polygon is not None:
            _plot_polygon(ax, polygon)
        else:
            logger.debug("No polygon provided; plotting points only")

        if gcp_xy:
            gx, gy = zip(*gcp_xy)
            ax.scatter(gx, gy, s=22, alpha=0.95, label="GCPs", marker="o")
        if vcp_xy:
            vx, vy = zip(*vcp_xy)
            ax.scatter(vx, vy, s=24, alpha=0.95, label="VCPs", marker="^")

        ax.set_xlabel("Easting (m)" if want_utm else "Longitude")
        ax.set_ylabel("Northing (m)" if want_utm else "Latitude")
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.6)
        ax.legend()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

        summary = f"GCP={len(gcp_xy)}, VCP={len(vcp_xy)}"
        logger.info("Saved overlay diagnostics to %s (%s)", output_path, summary)
        return {
            "image_path": output_path,
            "gcp_count": len(gcp_xy),
            "vcp_count": len(vcp_xy),
            "summary": summary
        }
    except Exception as e:
        logger.error("Overlay diagnostics failed: %s", e)
        return {"image_path": None, "gcp_count": 0, "vcp_count": 0, "summary": f"failed: {e}"}
