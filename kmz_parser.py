# kmz_parser.py
from __future__ import annotations
import os
import zipfile
import tempfile
import shutil
import logging
from typing import List, Tuple, Dict, Optional

import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import unary_union
from shapely.validation import make_valid

# Optional parser
try:
    from fastkml import kml as fastkml
except Exception:
    fastkml = None

_LOGGER = logging.getLogger(__name__)
if not _LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _extract_kml_from_kmz(path: str) -> Tuple[str, Optional[str]]:
    """
    Returns (kml_file_path, temp_dir). Caller should delete temp_dir if not None.
    Accepts .kmz or .kml path.
    """
    _, ext = os.path.splitext(path.lower())
    if ext == ".kml":
        return path, None
    if ext != ".kmz":
        raise ValueError(f"Unsupported file extension: {ext}. Expected .kmz or .kml")

    temp_dir = tempfile.mkdtemp(prefix="uas_kmz_")
    try:
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(temp_dir)
        for root, _, files in os.walk(temp_dir):
            for f in files:
                if f.lower().endswith(".kml"):
                    return os.path.join(root, f), temp_dir
    except Exception as exc:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to extract KMZ: {exc}") from exc

    shutil.rmtree(temp_dir, ignore_errors=True)
    raise RuntimeError("No .kml found inside the KMZ archive")


def _validate_coords_lonlat(coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Remove duplicate coords and normalize to floats."""
    if not coords:
        return []
    seen = set()
    out = []
    for lon, lat in coords:
        key = (round(float(lon), 8), round(float(lat), 8))
        if key not in seen:
            seen.add(key)
            out.append((float(lon), float(lat)))
    return out


def _polygon_from_coords(coords: List[Tuple[float, float]]) -> Optional[Polygon]:
    coords = _validate_coords_lonlat(coords)
    if len(coords) < 3:
        return None
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty or poly.area == 0:
        return None
    return poly


def _buffer_linestring(coords: List[Tuple[float, float]], width_deg: float = 1e-4) -> Optional[Polygon]:
    coords = _validate_coords_lonlat(coords)
    if len(coords) < 2:
        return None
    ls = LineString(coords)
    poly = ls.buffer(width_deg)
    if not poly.is_valid:
        poly = make_valid(poly)
    if poly.is_empty or poly.area == 0:
        return None
    return poly


def _parse_coords_text(text: str) -> List[Tuple[float, float]]:
    """
    KML <coordinates> content:
      'lon,lat[,alt] lon,lat[,alt] ...'
    """
    coords: List[Tuple[float, float]] = []
    if not text:
        return coords
    for token in text.strip().split():
        parts = token.split(",")
        if len(parts) >= 2:
            try:
                lon, lat = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            if -180 <= lon <= 180 and -90 <= lat <= 90:
                coords.append((lon, lat))
    return coords


def _parse_kml_xml(path: str) -> List[Dict]:
    """
    Robust, dependency-free KML parser (outer ring only + buffered lines).
    Returns list of dicts with shapely geometries.
    """
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    tree = ET.parse(path)
    root = tree.getroot()

    results: List[Dict] = []

    # Walk all Placemarks
    for pm in root.findall(".//kml:Placemark", ns):
        name_el = pm.find("kml:name", ns)
        name = name_el.text if name_el is not None else f"Placemark_{len(results)}"

        # Prefer Polygon
        poly_el = pm.find(".//kml:Polygon", ns)
        if poly_el is not None:
            outer_el = poly_el.find(".//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns)
            # (Optional) We could parse holes via innerBoundaryIs, but most AOIs don't use them.
            if outer_el is not None and outer_el.text:
                coords = _parse_coords_text(outer_el.text)
                shp = _polygon_from_coords(coords)
                if shp is not None and not shp.is_empty:
                    cx, cy = shp.centroid.x, shp.centroid.y
                    results.append({
                        "name": name,
                        "geometry": shp,
                        "coords": list(shp.exterior.coords),
                        "centroid": (cx, cy)
                    })
                    continue  # next Placemark

        # Fallback: LineString → thin polygon
        ls_el = pm.find(".//kml:LineString/kml:coordinates", ns)
        if ls_el is not None and ls_el.text:
            coords = _parse_coords_text(ls_el.text)
            shp = _buffer_linestring(coords)
            if shp is not None and not shp.is_empty:
                cx, cy = shp.centroid.x, shp.centroid.y
                results.append({
                    "name": name,
                    "geometry": shp,
                    "coords": list(shp.exterior.coords),
                    "centroid": (cx, cy)
                })
                continue

        # (Optional) MultiGeometry → pick first polygonal thing we can make
        mg_el = pm.find(".//kml:MultiGeometry", ns)
        if mg_el is not None:
            polys: List[Polygon] = []
            for subpoly in mg_el.findall(".//kml:Polygon", ns):
                outer_el = subpoly.find(".//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", ns)
                if outer_el is not None and outer_el.text:
                    coords = _parse_coords_text(outer_el.text)
                    shp = _polygon_from_coords(coords)
                    if shp is not None and not shp.is_empty:
                        polys.append(shp)
            if polys:
                merged = unary_union(polys)
                if not merged.is_empty:
                    if isinstance(merged, MultiPolygon):
                        # Keep largest piece
                        shp = max(merged.geoms, key=lambda g: g.area)
                    else:
                        shp = merged
                    cx, cy = shp.centroid.x, shp.centroid.y
                    results.append({
                        "name": name,
                        "geometry": shp,
                        "coords": list(shp.exterior.coords),
                        "centroid": (cx, cy)
                    })

    if not results:
        raise RuntimeError("No valid geometries found by XML parser")
    return results


def _parse_with_fastkml(path: str) -> List[Dict]:
    """Optional: use fastkml if present. Handles 'features' as list or method."""
    if not fastkml:
        raise RuntimeError("fastkml not available")
    with open(path, "r", encoding="utf-8") as f:
        kml_text = f.read()
    k = fastkml.KML()
    k.from_string(kml_text.encode("utf-8"))

    def _features(obj):
        feats = getattr(obj, "features", None)
        if callable(feats):
            return list(feats())
        if feats is None:
            return []
        # some versions expose a list directly
        return list(feats)

    out: List[Dict] = []

    def _walk(features, parent_name=""):
        for feat in features:
            name = getattr(feat, "name", None) or parent_name or f"Feature_{len(out)}"
            geom = getattr(feat, "geometry", None)
            if geom is None:
                _walk(_features(feat), name)
                continue

            shp = None
            try:
                # fastkml geom is usually shapely already
                if hasattr(geom, "geom_type"):
                    if geom.geom_type == "Polygon":
                        shp = _polygon_from_coords(list(geom.exterior.coords))
                    elif geom.geom_type == "MultiPolygon":
                        parts = [_polygon_from_coords(list(p.exterior.coords)) for p in geom.geoms]
                        parts = [p for p in parts if p is not None]
                        if parts:
                            shp = unary_union(parts)
                    elif geom.geom_type == "LineString":
                        shp = _buffer_linestring(list(geom.coords))
                else:
                    # fallback: try to treat it as polygon-like
                    if hasattr(geom, "exterior"):
                        shp = _polygon_from_coords(list(geom.exterior.coords))
            except Exception as exc:
                _LOGGER.debug("fastkml geometry adapt failed: %s", exc)

            if shp is not None and not shp.is_empty:
                cx, cy = shp.centroid.x, shp.centroid.y
                out.append({
                    "name": name,
                    "geometry": shp,
                    "coords": list(shp.exterior.coords),
                    "centroid": (cx, cy)
                })

            _walk(_features(feat), name)

    _walk(_features(k))
    if not out:
        raise RuntimeError("No valid geometries found by fastkml parser")
    return out


def parse_kml_geometries(kml_file: str) -> List[Dict]:
    """
    Prefer our dependency-free XML parser; if it fails, try fastkml (if installed).
    """
    # 1) XML-first (no external deps)
    try:
        return _parse_kml_xml(kml_file)
    except Exception as exc:
        _LOGGER.warning("XML parser failed: %s", exc)

    # 2) fastkml (optional)
    if fastkml:
        try:
            return _parse_with_fastkml(kml_file)
        except Exception as exc:
            _LOGGER.warning("fastkml parser failed: %s", exc)

    raise RuntimeError("All KML parsers failed")


def extract_geometries_from_kmz(kmz_path: str, as_geojson: bool = False) -> List[Dict]:
    """
    Extract geometries from KMZ/KML as a list of dicts:
    [{'name','geometry','coords','centroid'}], coords are (lon, lat).
    """
    temp_dir = None
    try:
        kml_file, temp_dir = _extract_kml_from_kmz(kmz_path)
        geoms = parse_kml_geometries(kml_file)
        if as_geojson:
            out = []
            for gd in geoms:
                out.append({
                    "type": "Polygon",
                    "coordinates": [list(gd["geometry"].exterior.coords)]
                })
            return out
        return geoms
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def extract_geometries_from_kmz_or_kml(path: str, as_geojson: bool = False) -> List[Dict]:
    """Compat helper (same as extract_geometries_from_kmz)."""
    return extract_geometries_from_kmz(path, as_geojson=as_geojson)


def extract_polygon_from_kmz(kmz_path: str) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    """Returns (coords, centroid) for the first polygon found (lon, lat)."""
    geoms = extract_geometries_from_kmz(kmz_path)
    polys = [g for g in geoms if isinstance(g["geometry"], (Polygon, MultiPolygon))]
    if not polys:
        raise ValueError("No polygon geometry found in KMZ/KML")
    g = polys[0]
    return list(g["geometry"].exterior.coords), g["centroid"]
