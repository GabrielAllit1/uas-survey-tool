# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['bootstrap_main.py'],
    pathex=[],
    binaries=[],
    datas=[('modern_ui.py', '.'), ('main_logic.py', '.'), ('proj_env.py', '.'), ('set_gdal_env.py', '.'), ('C:\\ProgramData\\miniconda3\\envs\\uas_survey_tool_v2\\Library\\share\\gdal', 'gdal'), ('C:\\ProgramData\\miniconda3\\envs\\uas_survey_tool_v2\\Library\\share\\proj', 'proj'), ('C:\\ProgramData\\miniconda3\\envs\\uas_survey_tool_v2\\Library\\bin', 'Library/bin')],
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
