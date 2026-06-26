# UAS_Survey_Tool_v2.spec
# PyInstaller 6.15+ compatible

import os, sys, glob
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

# --- Resolve project root (works even if __file__ is undefined) -------------
try:
    PRJ = Path(__file__).resolve().parent
except NameError:
    PRJ = Path.cwd()

# Entry point (you’ve been launching the app via bootstrap_main.py)
entry_script = str(PRJ / "bootstrap_main.py")

# --- Force DLL search to our UAS env only -----------------------------------
ENV = Path(os.environ.get("CONDA_PREFIX", r"C:\ProgramData\miniconda3\envs\uas_survey_tool_v2"))

def _rebuild_path(env: Path):
    bad_tokens = ["arbor_analyzer"]  # exclude foreign envs
    keep = []
    prepend = [
        str(env),
        str(env / "Library" / "mingw-w64" / "bin"),
        str(env / "Library" / "usr" / "bin"),
        str(env / "Library" / "bin"),
        str(env / "Scripts"),
        str(env / "bin"),
        str(env / "Lib" / "site-packages" / "PyQt6" / "Qt6" / "bin"),
    ]
    existing = os.environ.get("PATH", "").split(";")
    for p in existing:
        if p and all(bt.lower() not in p.lower() for bt in bad_tokens):
            keep.append(p)
    ordered = []
    for p in (prepend + keep):
        if p and (p not in ordered):
            ordered.append(p)
    return ";".join(ordered)

os.environ["PATH"] = _rebuild_path(ENV)

# Register critical DLL dirs for dependency resolver
for p in [
    ENV / "Library" / "bin",
    ENV / "Lib" / "site-packages" / "PyQt6" / "Qt6" / "bin",
]:
    p = str(p)
    if os.path.isdir(p):
        try:
            os.add_dll_directory(p)
        except Exception:
            pass
# ---------------------------------------------------------------------------

# -------------------------- Data & binary helpers --------------------------
datas = []
binaries = []

def add_dir_as_datas(src: Path, target_rel: str):
    if src.is_dir():
        datas.append((str(src), target_rel))

def add_dlls(src: Path, target_rel: str):
    if src.is_dir():
        for dll in src.glob("*.dll"):
            binaries.append((str(dll), target_rel))

# --- Include GDAL/PROJ share trees (conda-style layout) --------------------
gdal_share = ENV / "Library" / "share" / "gdal"
proj_share = ENV / "Library" / "share" / "proj"
add_dir_as_datas(gdal_share, "Library/share/gdal")
add_dir_as_datas(proj_share, "Library/share/proj")

# --- Include full Library/bin DLL set (GDAL, GEOS, PROJ, etc.) -------------
lib_bin = ENV / "Library" / "bin"
add_dlls(lib_bin, "Library/bin")

# --- Include Qt6 runtime bin (WebEngineProcess, etc.) -----------------------
qt6_bin = ENV / "Lib" / "site-packages" / "PyQt6" / "Qt6" / "bin"
add_dlls(qt6_bin, "PyQt6/Qt6/bin")

# ---------------------------- Project assets --------------------------------
# folders you likely want at runtime (non-code assets)
for folder in [
    PRJ / "assets",
    PRJ / "proj",
    PRJ / "dem_cache",
    PRJ / "dsm_cache",
    PRJ / "logs",
    PRJ / "dist_release",   # harmless if missing
]:
    if folder.exists():
        datas.append((str(folder), folder.name))

# individual files (if present)
for f in [
    "splash.png",
    "app_icon.ico",
    "check_icon.png",
    "fibonacci_icon.ico",
    "fractal_icon.ico",
    "GCP Pattern Map.png",
    "styles.qss",
    "Quick_Start_Guide.pdf",
    "about.txt",
    "license.txt",
    "issues.json",
    "opentopography_api.txt",
    "app.manifest",  # only as data; DO NOT pass to EXE(version=...) (that expects VSVersionInfo, not XML)
]:
    p = PRJ / f
    if p.exists():
        datas.append((str(p), "."))

# ------------------------- Third‑party hook data ----------------------------
# numpy/mpl/reportlab etc. (safe & small)
datas += collect_data_files("matplotlib", include_py_files=False)
datas += collect_data_files("certifi", include_py_files=False)
datas += collect_data_files("reportlab", include_py_files=False)
datas += collect_data_files("pyproj", include_py_files=False)
datas += collect_data_files("shapely", include_py_files=False)
datas += collect_data_files("rasterio", include_py_files=False)
datas += collect_data_files("fiona", include_py_files=False)
datas += collect_data_files("osgeo", include_py_files=False)

# ----------------------------- Hidden imports -------------------------------
hidden = []

# Rasterio & friends (explicit internal modules often missed)
hidden += collect_submodules("rasterio")
hidden += [
    "rasterio.mask",
    "rasterio.sample",
    "rasterio.features",
    "rasterio.warp",
    "rasterio.merge",
    "rasterio.plot",
    "rasterio.vrt",
]

# GDAL/OGR/OSR shim modules
hidden += ["osgeo", "osgeo.gdal", "osgeo.ogr", "osgeo.osr", "osgeo.gdalnumeric"]

# PyQt6 (incl. WebEngine)
hidden += collect_submodules("PyQt6")
hidden += collect_submodules("PyQt6.QtWebEngineCore")
hidden += collect_submodules("PyQt6.QtWebEngineWidgets")
hidden += collect_submodules("PyQt6.QtWebChannel")
hidden += collect_submodules("PyQt6.QtTest")
hidden += collect_submodules("PyQt6.QtXml")
hidden += collect_submodules("PyQt6.QtTextToSpeech")
hidden += collect_submodules("PyQt6.QtWebSockets")

# Geo stack helpers
hidden += collect_submodules("pyproj")
hidden += collect_submodules("shapely")
hidden += collect_submodules("fiona")

# -------------------------- Your local modules ------------------------------
# Every .py in project root you listed (force include)
local_modules = [
    # core app
    "bootstrap_main", "main", "modern_ui", "main_logic",
    # geo / pipeline
    "gcp_generator", "overlay_diagnostics", "overlay_generator", "overlay_map", "overlay_map_generator",
    "kmz_parser", "parse_kml",
    "elevation_data", "elevation_service", "elevation_utils",
    "dem_downloader", "dsm_manager", "dsm_diagnostics", "terrain_analyzer",
    "survey_diagnostics", "flight_parameters_calculator", "virtual_dem_generator",
    # report/export
    "pdf_report", "report_generator", "export_manager",
    # utils/misc
    "api_key_manager", "crs_utils", "utils", "math_utils", "area_utils",
    "filter_coordinates", "vegetation_filter",
    "icon_handler", "safe_fmt", "data_dictionary",
    "mathtoolbox_complete", "mathtoolbox_refined", "math_tool_box", "unified_mathtoolbox",
    "geometry_type_checker", "find_wrong_latlon", "fix_latlon_order",
    "proj_env", "set_env_vars", "set_gdal_env",
]
hidden += local_modules

# ----------------------------- Runtime hooks --------------------------------
# rth_path_sanitize.py (we’ll create this file below), plus your set_gdal_env.py
runtime_hooks = []
rth_path = PRJ / "rth_path_sanitize.py"
if rth_path.exists():
    runtime_hooks.append(str(rth_path))
set_gdal = PRJ / "set_gdal_env.py"
if set_gdal.exists():
    runtime_hooks.append(str(set_gdal))

# ----------------------------- Build steps ----------------------------------
a = Analysis(
    [entry_script],
    pathex=[str(PRJ)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],               # you can add a custom hooks dir here if you make one
    runtime_hooks=runtime_hooks,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="UAS_Survey_Tool_v2",
    icon=str(PRJ / "app_icon.ico") if (PRJ / "app_icon.ico").exists() else None,
    console=False,                      # GUI app
    disable_windowed_traceback=True,    # cleaner error popup
    # debug=['imports'],                # uncomment while diagnosing missing imports
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="UAS_Survey_Tool_v2",
)
