import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files('osgeo') + collect_data_files('osgeo.gdal') + collect_data_files('osgeo.ogr') + collect_data_files('osgeo.osr')
datas += collect_data_files('gdal', include_py_files=True)
env_root = Path(os.environ.get("CONDA_PREFIX", sys.prefix))
for src, target in [
    (env_root / "Library" / "share" / "gdal", "gdal"),
    (env_root / "Library" / "share" / "proj", "proj"),
    (env_root / "Library" / "bin", "Library/bin"),
]:
    if src.exists():
        datas.append((str(src), target))
hiddenimports = collect_submodules('rasterio') + ['osgeo', 'osgeo.gdal', 'osgeo.ogr', 'osgeo.osr']
