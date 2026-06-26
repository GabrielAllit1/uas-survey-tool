import os
import logging
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'logs', 'uas_survey_tool.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def generate_report(
    output_path: str,
    polygon_coords: list,
    gcp_points: list,
    veri_points: list,
    spacing: int,
    layout_mode: str,
    area_acres: float,
    elevation_data: dict,
    dsm_summary: str,
    overlay_image_path: str,
    flight_timings: dict = None,
    pattern_score: float = 0.0
):
    """
    Generate a PDF report for the survey plan.

    Args:
        output_path (str): Path to save the PDF.
        polygon_coords (list): List of (lon, lat) tuples.
        gcp_points (list): List of GCP dictionaries.
        veri_points (list): List of VCP dictionaries.
        spacing (int): GCP spacing in meters.
        layout_mode (str): Layout mode (e.g., 'grid').
        area_acres (float): Survey area in acres.
        elevation_data (dict): Elevation statistics.
        dsm_summary (str): DSM analysis summary.
        overlay_image_path (str): Path to overlay map image.
        flight_timings (dict, optional): Flight timing data.
        pattern_score (float): Pattern score for points.
    """
    try:
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("UAS Survey Tool Report", styles['Title']))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Area: {area_acres:.2f} acres", styles['Normal']))
        story.append(Paragraph(f"Layout Mode: {layout_mode}", styles['Normal']))
        story.append(Paragraph(f"GCP Spacing: {spacing}m", styles['Normal']))
        story.append(Paragraph(f"Pattern Score: {pattern_score:.2f}", styles['Normal']))
        story.append(Paragraph(f"Elevation Data: Min={elevation_data.get('min', 0):.2f}m, Max={elevation_data.get('max', 0):.2f}m, Mean={elevation_data.get('mean', 0):.2f}m", styles['Normal']))
        story.append(Paragraph(f"DSM Summary: {dsm_summary}", styles['Normal']))
        story.append(Spacer(1, 12))

        story.append(Paragraph(f"GCPs: {len(gcp_points)}", styles['Heading2']))
        for pt in gcp_points[:5]:  # Limit for brevity
            story.append(Paragraph(f"{pt['name']}: ({pt['easting']:.2f}, {pt['northing']:.2f}), Elev: {pt['elevation']:.2f}m", styles['Normal']))
        story.append(Paragraph(f"Verification Points: {len(veri_points)}", styles['Heading2']))
        for pt in veri_points[:5]:
            story.append(Paragraph(f"{pt['name']}: ({pt['easting']:.2f}, {pt['northing']:.2f}), Elev: {pt['elevation']:.2f}m", styles['Normal']))
        story.append(Spacer(1, 12))

        if flight_timings:
            story.append(Paragraph("Flight Timings:", styles['Heading2']))
            for key, value in flight_timings.items():
                story.append(Paragraph(f"{key}: {value}", styles['Normal']))
            story.append(Spacer(1, 12))

        if os.path.exists(overlay_image_path):
            story.append(Image(overlay_image_path, width=400, height=300))
            logging.debug(f"Included overlay image: {overlay_image_path}")

        doc.build(story)
        logging.info(f"Generated report at {output_path}")

    except Exception as e:
        logging.error(f"Report generation failed: {str(e)}")