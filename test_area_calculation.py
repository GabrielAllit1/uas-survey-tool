import logging
from kmz_parser import extract_geometries_from_kmz_or_kml
from shapely.geometry import MultiPolygon, Polygon
from area_utils import calculate_area_acres

# Setup basic logging to console at DEBUG level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def select_largest_polygon(geom):
    if isinstance(geom, MultiPolygon):
        largest = max(geom.geoms, key=lambda p: p.area)
        logging.debug(f"Selected largest polygon from MultiPolygon with area: {largest.area}")
        return largest
    elif isinstance(geom, Polygon):
        return geom
    else:
        logging.warning(f"Geometry type {geom.geom_type} is not Polygon or MultiPolygon")
        return None

def main():
    test_file = r"C:/Users/gabri/OneDrive/Documents/MasTec/KMZ/kmzpack/2025-04-25_WDK_AOI.kml"  # Change to your test file path

    print(f"Loading AOI geometries from: {test_file}")
    geoms = extract_geometries_from_kmz_or_kml(test_file)

    print(f"Extracted {len(geoms)} geometries:")
    for idx, gdict in enumerate(geoms):
        name = gdict.get("name", "Unnamed")
        geom = gdict.get("geometry")
        print(f" {idx+1}. Name: {name}, Type: {geom.geom_type}")

    # Select first polygon or largest polygon
    polygon = None
    for gdict in geoms:
        candidate = select_largest_polygon(gdict["geometry"])
        if candidate:
            polygon = candidate
            break

    if polygon is None:
        print("No valid polygon found in geometries!")
        return

    # Get polygon exterior coordinates as (lon, lat) tuples
    coords = list(polygon.exterior.coords)
    print(f"Polygon exterior has {len(coords)} coordinates")

    # Run area calculation
    area_m2, area_acres, fam_scale, pattern_score = calculate_area_acres(coords)
    print(f"Calculated polygon area: {area_m2:.2f} m² ({area_acres:.4f} acres)")
    print(f"FAM scale factor: {fam_scale:.4f}")
    print(f"Pattern score: {pattern_score:.4f}")

if __name__ == "__main__":
    main()
