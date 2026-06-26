import os
import logging
import shutil
from pathlib import Path
from pyproj import Proj, datadir

logger = logging.getLogger(__name__)

def initialize_proj_env():
    """Set up PROJ and GDAL environment variables and context."""
    try:
        # Base path for Conda environment
        conda_env_path = os.environ.get('CONDA_PREFIX', '')
        if not conda_env_path:
            logger.warning("CONDA_PREFIX not set, attempting to find PROJ database")
            conda_env_path = os.path.dirname(os.path.dirname(os.__file__))

        # Set PROJ_LIB and pyproj data directory
        proj_lib = os.path.join(conda_env_path, 'Library', 'share', 'proj')
        if os.path.exists(proj_lib) and os.path.isfile(os.path.join(proj_lib, 'proj.db')):
            os.environ['PROJ_LIB'] = proj_lib
            datadir.set_data_dir(proj_lib)  # Explicitly set pyproj data directory
            logger.info(f"PROJ_LIB set to {proj_lib}")
        else:
            logger.warning(f"PROJ database not found at {proj_lib}")
            possible_paths = [
                Path(conda_env_path) / 'share' / 'proj',
                Path(conda_env_path) / 'Library' / 'share' / 'proj',
                Path('/usr') / 'share' / 'proj',
                Path('/usr') / 'local' / 'share' / 'proj',
            ]
            for path in possible_paths:
                if path.exists() and (path / 'proj.db').exists():
                    os.environ['PROJ_LIB'] = str(path)
                    datadir.set_data_dir(str(path))
                    logger.info(f"Fallback PROJ_LIB set to {path}")
                    break
            else:
                logger.error("No valid PROJ database path found")

        # Remove conflicting PROJ path in pyproj
        conflicting_path = os.path.join(conda_env_path, 'lib', 'site-packages', 'pyproj', 'proj_dir', 'share', 'proj')
        if os.path.exists(conflicting_path):
            logger.warning(f"Removing conflicting PROJ path: {conflicting_path}")
            try:
                shutil.rmtree(conflicting_path)
                logger.info(f"Successfully removed conflicting PROJ path: {conflicting_path}")
            except Exception as e:
                logger.error(f"Failed to remove conflicting PROJ path {conflicting_path}: {e}")

        # Set GDAL_DATA
        gdal_data = os.path.join(conda_env_path, 'Library', 'share', 'gdal')
        if os.path.exists(gdal_data):
            os.environ['GDAL_DATA'] = gdal_data
            logger.info(f"GDAL_DATA set to {gdal_data}")
        else:
            logger.warning(f"GDAL data not found at {gdal_data}")

        # Disable PROJ network to avoid online queries
        os.environ['PROJ_NETWORK'] = 'OFF'
        logger.info("PROJ_NETWORK set to OFF")

        # Verify PROJ context
        try:
            proj = Proj('EPSG:4326')
            logger.info(f"PROJ context verified with EPSG:4326: {proj}")
        except Exception as e:
            logger.error(f"Failed to verify PROJ context: {e}")

        logger.info(f"proj_env initialized. PROJ_LIB={os.environ.get('PROJ_LIB', 'Not set')} | "
                    f"GDAL_DATA={os.environ.get('GDAL_DATA', 'Not set')} | "
                    f"PROJ_NETWORK={os.environ.get('PROJ_NETWORK', 'Not set')}")
    except Exception as e:
        logger.error(f"Failed to initialize proj_env: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize_proj_env()