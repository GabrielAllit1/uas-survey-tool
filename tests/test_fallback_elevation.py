import os
import pytest
from shapely.geometry import Polygon
from main_logic import MainLogic
from kmz_parser import extract_geometries_from_kmz
from pyproj import CRS
from unittest.mock import patch
from elevation_data import get_srtm_elevation_bulk

@pytest.fixture(autouse=True)
def no_files(tmp_path, monkeypatch):
    # Mock os.path.exists to return False for DEM/DSM paths
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    # Mock get_srtm_elevation_bulk to return valid elevations
    monkeypatch.setattr(
        "elevation_data.get_srtm_elevation_bulk",
        lambda coords, retry, delay, dataset: ([300.0] * len(coords), 300.0)
    )
    return tmp_path

def test_generate_plan_without_dem_dsm(tmp_path):
    ml = MainLogic(parent=None, cuda_enabled=False)
    # Simulate KMZ loading with realistic UTM coordinates
    ml.polygon_coords = [(-100.0, 40.0), (-100.0, 40.001), (-99.999, 40.001), (-99.999, 40.0)]
    ml.polygon = Polygon(ml.polygon_coords)
    ml.centroid = (-99.9995, 40.0005)
    ml.polygon_area_acres = 0.1
    ml.utm_crs = CRS.from_epsg(32614)  # UTM Zone 14N
    ml.polygon_utm = Polygon([(500000, 4430000), (500000, 4430111), (500111, 4430111), (500111, 4430000)])
    ml.last_kmz_path = str(tmp_path / "test.kmz")

    # Generate plan with no DEM/DSM and auto-download disabled
    gcp_pts, vcp_pts, elev_data = ml.generate_gcp_plan(
        auto_download_dem=False,
        auto_download_dsm=False,
        user_spacing=50.0,
        layout_mode="grid",
        sequence_type="fibonacci",
        weights=None,
        dem_path=None,
        dsm_path=None,
        density_factor=1.0
    )

    # Assertions
    assert isinstance(elev_data, dict)
    assert "profile" in elev_data
    assert len(elev_data["profile"]) > 0
    assert elev_data["min"] >= 0
    assert elev_data["mean"] >= 0
    assert len(gcp_pts) > 0 or len(vcp_pts) > 0
    assert all(pt.get("elevation", 0) >= 0 for pt in gcp_pts + vcp_pts)