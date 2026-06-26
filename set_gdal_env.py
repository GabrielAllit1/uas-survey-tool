# set_gdal_env.py  — runtime hook for PyInstaller
import logging, os, sys
from pathlib import Path

# ---------- logging ----------
LOG_DIR = Path.home() / "UAS_Survey_Tool_Logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "frozen_env_check.log"
logger = logging.getLogger("GDAL_ENV")
logger.setLevel(logging.DEBUG)
logger.propagate = False
if not logger.handlers:
    fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)

# ---------- resolve bundle base ----------
BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
LIB = BASE / "Library"
LIB_BIN = LIB / "bin"
GDAL_SHARE = LIB / "share" / "gdal"
PROJ_SHARE = LIB / "share" / "proj"

def _prepend_path(p: Path):
    p = str(p)
    cur = os.environ.get("PATH", "")
    if p not in cur.split(os.pathsep):
        os.environ["PATH"] = p + os.pathsep + cur

# Prefer the bundled dirs when frozen; otherwise let proj_env fill them in
if getattr(sys, "frozen", False):
    if GDAL_SHARE.exists():
        os.environ["GDAL_DATA"] = str(GDAL_SHARE)
    if PROJ_SHARE.exists():
        os.environ["PROJ_LIB"] = str(PROJ_SHARE)
    # Make sure DLLs resolve
    if LIB_BIN.exists():
        _prepend_path(LIB_BIN)
    # Helpful flags
    os.environ.setdefault("USE_PATH_FOR_GDAL_DATA", "YES")
    os.environ.setdefault("CPL_ZIP_ENCODING", "UTF-8")
    # Disable network grid fetches in offline/customer machines; enable if you rely on remote grids
    os.environ.setdefault("PROJ_NETWORK", "OFF")
else:
    # Not frozen -> fall back to your standard environment config
    try:
        import proj_env  # this should set PROJ_LIB / GDAL_DATA for dev runs
        logger.info("proj_env.configure() applied in dev mode.")
    except Exception as e:
        logger.exception("proj_env import/config failed: %s", e)

# ---- sanity dump ----
for k in ("PROJ_LIB", "GDAL_DATA", "PROJ_NETWORK"):
    logger.info("%s = %s", k, os.environ.get(k))
logger.info("PATH head: %s", os.environ.get("PATH", "")[:240])
logger.info("Frozen=%s  _MEIPASS=%s", getattr(sys, "frozen", False), getattr(sys, "_MEIPASS", None))
