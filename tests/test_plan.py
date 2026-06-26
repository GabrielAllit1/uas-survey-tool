import pytest
from shapely.geometry import Polygon
from main_logic import MainLogic
from pyproj import CRS
from unittest.mock import Mock, patch
from elevation_data import get_srtm_elevation_bulk

@pytest.fixture
def main_logic():
    ml = MainLogic(parent=None, cuda_enabled=False)
    ml.polygon_coords = [(-100.0, 40.0), (-100.0, 40.001), (-99.999, 40.001), (-99.999, 40.0)]
    ml.polygon = Polygon(ml.polygon_coords)
    ml.centroid = (-99.9995, 40.0005)
    ml.polygon_area_acres = 0.1
    ml.utm_crs = CRS.from_epsg(32614)  # UTM Zone 14N
    ml.polygon_utm = Polygon([(500000, 4430000), (500000, 4430111), (500111, 4430111), (500111, 4430000)])
    return ml

def test_plan_generation(main_logic):
    with patch("elevation_data.get_srtm_elevation_bulk", return_value=([300.0] * 4, 300.0)):
        gcp_pts, vcp_pts, elev_data = main_logic.generate_gcp_plan(
            user_spacing=50.0,
            layout_mode="grid",
            sequence_type="fibonacci",
            weights=[0.25, 0.25, 0.25, 0.25],
            dem_path=None,
            dsm_path=None,
            auto_download_dem=False,
            auto_download_dsm=False,
            density_factor=1.0
        )
        assert isinstance(gcp_pts, list)
        assert isinstance(vcp_pts, list)
        assert isinstance(elev_data, dict)
        assert "profile" in elev_data
        assert len(gcp_pts) + len(vcp_pts) > 0
        assert all(pt.get("elevation", 0) >= 0 for pt in gcp_pts + vcp_pts)
        assert all("name" in pt for pt in gcp_pts + vcp_pts)