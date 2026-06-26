import logging
import os

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.expanduser("~/UAS_Survey_Tool_Logs"), 'area_utils.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def calculate_area_acres(area_m2: float) -> float:
    """
    Convert area from square meters to acres.

    Args:
        area_m2 (float): Area in square meters.

    Returns:
        float: Area in acres.

    Raises:
        ValueError: If area_m2 is negative or invalid.
    """
    try:
        if not isinstance(area_m2, (int, float)) or area_m2 < 0:
            logger.error(f"Invalid area input: {area_m2}. Must be a non-negative number.")
            raise ValueError(f"Invalid area input: {area_m2}. Must be a non-negative number.")
        acres = area_m2 * 0.000247105  # 1 acre = 4046.86 m²
        logger.debug(f"Converted {area_m2:.2f} m² to {acres:.2f} acres")
        return acres
    except Exception as e:
        logger