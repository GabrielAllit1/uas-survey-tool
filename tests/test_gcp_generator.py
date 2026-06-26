import pytest
from shapely.geometry import Polygon
from gcp_generator import generate_gcp
from main_logic import MainLogic
from pyproj import CRS
from unittest.mock import Mock

@pytest.fixture
def main_logic():
    ml = Mock(spec=MainLogic)
    ml.crs = CRS.from_epsg(32614)  # UTM Zone 14N
    ml.get_elevation = Mock(return_value=291.0)
    ml.min_gcp_count = 15
    ml.max_gcp_count = 1000
    ml.min_ver_count = 25
    ml.max_ver_count = 1500
    ml.relax_filters = False
    return ml

@pytest.fixture
def aoi():
    coords = [(-100.0, 40.0), (-100.0, 40.1), (-99.9, 40.1), (-99.9, 40.0)]
    return Polygon(coords)

@pytest.mark.parametrize("pattern", [
    "grid", "triangular", "diamond", "fractal", "chaotic", "spiral"
])
def test_pattern_basic_metrics(pattern, main_logic, aoi):
    fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    kap_sequence = [1, 1, 3, 5, 9, 17, 31, 57, 105, 193]
    gcp_pts, vcp_pts, rej_gcp, rej_vcp = generate_gcp(
        polygon_wgs84=aoi,
        spacing=50.0,
        layout_mode=pattern,
        sequence_type="fibonacci",
        modulus=13,
        min_distance_factor=0.5,
        dem_path=None,
        dsm_path=None,
        fib_sequence=fib_sequence,
        kap_sequence=kap_sequence,
        main_logic=main_logic,
        initial_points=None,
        density_factor=1.0
    )
    assert isinstance(gcp_pts, list)
    assert isinstance(vcp_pts, list)
    assert isinstance(rej_gcp, list)
    assert isinstance(rej_vcp, list)
    assert len(gcp_pts) + len(vcp_pts) >= 0
    assert all(pt.get("elevation", 0) >= 0 for pt in gcp_pts + vcp_pts)
    assert all("name" in pt for pt in gcp_pts + vcp_pts)