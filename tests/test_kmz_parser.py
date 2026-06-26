import os
import zipfile
import tempfile
import pytest
from kmz_parser import extract_geometries_from_kmz, extract_geometries_from_kmz_or_kml
from shapely.geometry import Polygon, LineString

@pytest.fixture
def temp_kmz(tmp_path):
    kml_content = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <Placemark>
        <name>Test Polygon</name>
        <Polygon>
            <outerBoundaryIs>
                <LinearRing>
                    <coordinates>
                        -100,40,0 -100,40.1,0 -99.9,40.1,0 -99.9,40,0
                    </coordinates>
                </LinearRing>
            </outerBoundaryIs>
        </Polygon>
    </Placemark>
</Document>
</kml>"""
    kml_path = tmp_path / "test.kml"
    with open(kml_path, 'w') as f:
        f.write(kml_content)
    kmz_path = tmp_path / "test.kmz"
    with zipfile.ZipFile(kmz_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(kml_path, "doc.kml")
    return str(kmz_path)

def test_extract_geometries_from_kmz(temp_kmz):
    # Test default output (list of dicts)
    geoms = extract_geometries_from_kmz(temp_kmz)
    assert isinstance(geoms, list)
    assert len(geoms) > 0
    assert isinstance(geoms[0]["geometry"], (Polygon, LineString))
    assert "coords" in geoms[0]
    assert "name" in geoms[0]
    assert "pattern_score" in geoms[0]

    # Test GeoJSON output
    geoms_geojson = extract_geometries_from_kmz(temp_kmz, as_geojson=True)
    assert isinstance(geoms_geojson, list)
    assert len(geoms_geojson) > 0
    assert geoms_geojson[0]["type"] in ["Polygon", "LineString", "Point"]
    assert "coordinates" in geoms_geojson[0]

def test_extract_geometries_from_kmz_or_kml(temp_kmz):
    # Test KMZ
    geoms = extract_geometries_from_kmz_or_kml(temp_kmz)
    assert isinstance(geoms, list)
    assert len(geoms) > 0
    assert isinstance(geoms[0]["geometry"], (Polygon, LineString))
    assert "coords" in geoms[0]
    assert "name" in geoms[0]
    assert "pattern_score" in geoms[0]

    # Test GeoJSON output
    geoms_geojson = extract_geometries_from_kmz_or_kml(temp_kmz, as_geojson=True)
    assert isinstance(geoms_geojson, list)
    assert len(geoms_geojson) > 0
    assert geoms_geojson[0]["type"] in ["Polygon", "LineString", "Point"]
    assert "coordinates" in geoms_geojson[0]

    # Test KML (create a KML file)
    kml_path = temp_kmz.replace(".kmz", ".kml")
    with open(kml_path, 'w') as f:
        f.write("""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
    <Placemark>
        <name>Test Polygon</name>
        <Polygon>
            <outerBoundaryIs>
                <LinearRing>
                    <coordinates>
                        -100,40,0 -100,40.1,0 -99.9,40.1,0 -99.9,40,0
                    </coordinates>
                </LinearRing>
            </outerBoundaryIs>
        </Polygon>
    </Placemark>
</Document>
</kml>""")
    geoms_kml = extract_geometries_from_kmz_or_kml(kml_path)
    assert isinstance(geoms_kml, list)
    assert len(geoms_kml) > 0
    assert isinstance(geoms_kml[0]["geometry"], (Polygon, LineString))
    assert "coords" in geoms_kml[0]
    assert "name" in geoms_kml[0]
    assert "pattern_score" in geoms_kml[0]