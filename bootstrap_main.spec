# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

project_root = Path.cwd()
env_root = Path(os.environ.get("CONDA_PREFIX", sys.prefix))

gdal_share = env_root / "Library" / "share" / "gdal"
proj_share = env_root / "Library" / "share" / "proj"
library_bin = env_root / "Library" / "bin"

a = Analysis(
    ['bootstrap_main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('modern_ui.py', '.'),
        ('main_logic.py', '.'),
        ('proj_env.py', '.'),
        ('set_gdal_env.py', '.'),
    ] + ([(str(gdal_share), 'gdal')] if gdal_share.exists() else [])
      + ([(str(proj_share), 'proj')] if proj_share.exists() else [])
      + ([(str(library_bin), 'Library/bin')] if library_bin.exists() else []),
    hiddenimports=['rasterio', 'rasterio.sample', 'rasterio.mask', 'osgeo', 'osgeo.gdal', 'osgeo.ogr', 'osgeo.osr', 'PyQt6', 'geopandas', 'numpy', 'pyproj', 'shapely'],
    hookspath=['.'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='bootstrap_main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
