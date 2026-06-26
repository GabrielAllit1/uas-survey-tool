import statistics
import numpy as np
from unified_mathtoolbox import MathToolBox
from elevation_service import detect_elevation_cycles
import logging
import os

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.expanduser("~/UAS_Survey_Tool_Logs"), 'survey_diagnostics.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_survey_diagnostics(elevation_data):
    """
    Generates a human-readable terrain analysis summary from elevation data.
    Returns diagnostic string and pattern score.
    """
    if not elevation_data or len(elevation_data) < 2:
        logger.debug("No usable elevation data available for terrain analysis")
        return "⚠️ No usable elevation data available for terrain analysis.", 0.0

    min_elev = min(elevation_data)
    max_elev = max(elevation_data)
    relief = max_elev - min_elev
    mean_elev = sum(elevation_data) / len(elevation_data)
    std_dev = statistics.stdev(elevation_data) if len(elevation_data) > 1 else 0
    percent_variation = (relief / mean_elev * 100) if mean_elev else 0

    cycle_score = detect_elevation_cycles(elevation_data)
    toolbox = MathToolBox()
    pattern_score = toolbox.unified_detection(len(elevation_data), 0)
    complexity = cycle_score + pattern_score

    if complexity > 1.4 or relief > 20:
        rating = "❗ Significant terrain complexity"
        recommendation = "Fractal or Chaotic layouts strongly recommended."
    elif relief < 1:
        rating = "✅ Flat terrain"
        recommendation = "Grid or Triangular layouts are ideal."
    elif relief < 5 or complexity < 0.7:
        rating = "ℹ️ Mild terrain variation"
        recommendation = "Triangular or Spiral layouts preferred for better coverage."
    else:
        rating = "⚠️ Moderate relief"
        recommendation = "Spiral or Diamond layout for elevation changes."

    diagnostic = (
        f"{rating}\n"
        f"• Elevation range: {min_elev:.2f} m – {max_elev:.2f} m\n"
        f"• Total relief: {relief:.2f} m\n"
        f"• Std dev: {std_dev:.2f} m, Variability: {percent_variation:.1f}%\n"
        f"• Cycle Score: {cycle_score:.2f}, Pattern Score: {pattern_score:.2f}\n"
        f"→ {recommendation}"
    )

    logger.debug(f"Survey Diagnostics: {diagnostic}")
    return diagnostic, pattern_score