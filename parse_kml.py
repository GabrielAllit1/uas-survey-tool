# parse_kml.py
from __future__ import annotations

import io
import logging
import os
import zipfile
import xml.etree.ElementTree as ET
from typing import Iterable, List, Tuple, Optional

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(os.path.expanduser("~/UAS_Survey_Tool_Logs"), "uas_survey_tool.log"),
                encoding="utf-8",
            ),
            logging.StreamHandler(),
        ],
    )

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _looks_like_lon_lat(a: float, b: float) -> bool:
    return -180.0 <= a <= 180.0 and -90.0 <= b <= 90.0

def _looks_like_lat_lon(a: float, b: float) -> bool:
    return -90.0 <= a <= 90.0 and -180.0 <= b <= 180.0

def _parse_coord_token(tok: str) -> Tuple[float, float]:
    """
    Parse a single KML token:  'lon,lat' or 'lon,lat,alt'
    Some files incorrectly save 'lat,lon(,alt)'. Detect & swap.
    Always returns (lon, lat).
    """
    parts = tok.strip().split(",")
    if len(parts) < 2:
        raise ValueError(f"Invalid coordinate token: {tok!r}")
    a = float(parts[0])
    b = float(parts[1])

    if _looks_like_lon_lat(a, b):
        return (a, b)
    if _looks_like_lat_lon(a, b):
        logger.warning("Detected (lat,lon) ordering; auto-swapping to (lon,lat).")
        return (b, a)

    logger.warning(f"Out-of-range coord encountered; clamping best-effort: {tok!r}")
    a = max(min(a, 180.0), -180.0)
    b = max(min(b, 90.0), -90.0)
    return (a, b)

def _extract_coords(text: str) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for tok in text.strip().split():
        try:
            pts.append(_parse_coord_token(tok))
        except Exception as e:
            logger.warning(f"Skipping bad coordinate token {tok!r}: {e}")
    # drop consecutive duplicates
    out: List[Tuple[float, float]] = []
    last: Optional[Tuple[float, float]] = None
    for p in pts:
        if p != last:
            out.append(p)
        last = p
    return out

def _close_ring_if_needed(ring: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(ring) >= 3 and ring[0] != ring[-1]:
        ring = ring + [ring[0]]
    return ring

def _iter_coordinates_nodes(root: ET.Element) -> Iterable[ET.Element]:
    """
    Iterate over all <coordinates> nodes, regardless of namespace or geometry type.
    Supports Polygon (outer/inner), LineString, MultiGeometry, gx:Track (as lon,lat,alt in <gx:coord>).
    """
    # 1) Standard KML coordinates
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "coordinates":
            yield elem

    # 2) gx:Track coords (as <gx:coord>lon lat alt</gx:coord>)
    ns_gx = "{http://www.google.com/kml/ext/2.2}"
    for track in root.findall(f".//{ns_gx}Track"):
        for coord in track.findall(f"{ns_gx}coord"):
            txt = coord.text or ""
            parts = txt.strip().split()
            if len(parts) >= 2:
                lon = float(parts[0])
                lat = float(parts[1])
                # synthesize an equivalent <coordinates> node for uniform processing
                fake = ET.Element("coordinates")
                fake.text = f"{lon},{lat}"
                yield fake

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def parse_kml(kml_content: str) -> List[List[Tuple[float, float]]]:
    """
    Parse KML content and return list of geometries,
    each geometry as list of (lon, lat) tuples.
    - Polygons are closed if needed.
    - Holes are ignored (logged).
    - Auto-fixes (lat,lon) mistakes.
    - Supports MultiGeometry and gx:Track.
    """
    root = ET.fromstring(kml_content)

    # Collect raw rings/paths (outer rings first if Polygon)
    geoms: List[List[Tuple[float, float]]] = []
    for coords_node in _iter_coordinates_nodes(root):
        coords_text = coords_node.text or ""
        ring = _extract_coords(coords_text)
        ring = _close_ring_if_needed(ring)
        if len(ring) >= 2:
            geoms.append(ring)

    if not geoms:
        logger.warning("No coordinates found in KML content.")
    else:
        logger.debug(f"Parsed {len(geoms)} geometry coordinate sets from KML.")

    return geoms


def parse_kmz_or_kml(path: str) -> List[List[Tuple[float, float]]]:
    """
    Load KML/KMZ from disk and return list of (lon,lat) sequences.
    KMZ: reads doc.kml or the first *.kml inside the archive.
    """
    path = os.fspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.lower().endswith(".kmz"):
        with zipfile.ZipFile(path, "r") as zf:
            # Prefer doc.kml if present
            kml_name = None
            if "doc.kml" in zf.namelist():
                kml_name = "doc.kml"
            else:
                for n in zf.namelist():
                    if n.lower().endswith(".kml"):
                        kml_name = n
                        break
            if not kml_name:
                raise ValueError("KMZ has no KML inside.")
            with zf.open(kml_name) as fh:
                kml_bytes = fh.read()
        return parse_kml(kml_bytes.decode("utf-8", errors="ignore"))

    # Plain KML
    with io.open(path, "r", encoding="utf-8", errors="ignore") as f:
        return parse_kml(f.read())


def to_lonlat_tuples(geometries: Iterable[Iterable[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
    """
    Identity helper, retained for API parity in your codebase.
    Keeps outputs consistently (lon, lat).
    """
    return [list(ring) for ring in geometries]

