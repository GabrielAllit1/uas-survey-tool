import os
import pytest
from dem_downloader import DEMDownloader
from dsm_manager import get_dsm_elevation_profile

@pytest.fixture(autouse=True)
def clean_cache(tmp_path, monkeypatch):
    # point the cache to a temp dir so no files exist
    monkeypatch.setenv("HOME", str(tmp_path))
    yield

def test_dem_download_fails_returns_none(tmp_path):
    # give it a bogus KMZ path → download_dem should return None
    downloader = DEMDownloader()
    fake_kmz = str(tmp_path / "no_such.kmz")
    # ensure file doesn’t exist
    if os.path.exists(fake_kmz):
        os.remove(fake_kmz)
    out = downloader.download_dem(fake_kmz)
    assert out is None

def test_dsm_profile_without_raster(monkeypatch):
    # patch raster path to None and simulate a simple AOI coordinate list
    coords = [( -75.0,  40.0 ), ( -75.1, 40.1 ), ( -75.2, 40.2 )]
    # force os.path.exists to always say “no raster file”
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    elevs, cycle_score, pattern_score = get_dsm_elevation_profile(coords, raster_path=None)
    # we should get back a list of floats, one per coord
    assert len(elevs) == len(coords)
    # and none of them should be None
    assert all(isinstance(e, float) for e in elevs)
    # scores should be within [0,1]
    assert 0.0 <= cycle_score <= 1.0
    assert 0.0 <= pattern_score <= 1.0
