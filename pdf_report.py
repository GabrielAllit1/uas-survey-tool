from __future__ import annotations

"""
PDF Report Generator — UAS Survey Tool v2.0

This module builds a polished, professional PDF report that rivals top-tier vendors.
Key enhancements vs. prior version:
  • Executive Summary block with KPIs and compliance targets
  • Dual methodology section: HYBRID method (rotational shots) + STANDARD method
    - Standard: 180s static GCP (no rod rotation), 30s static VCP (no rod rotation)
  • Comprehensive metrics table, terrain analysis, overlay map, elevation charts
  • Optional 3D DSM/DEM rendition
  • QA/QC accuracy table (RMSE/ME/SD) and acceptance criteria highlight
  • Rejected points summary (if provided)
  • Coverage heatmap hook (optional image path)
  • Footer with versioning and sign-off section
  • Safe fallbacks so the report never crashes on missing optional data

Inputs remain backward-compatible with the previous pdf_report.py and accept
several new optional parameters for extra polish.
"""

import os
import tempfile
import datetime
import logging
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

try:
    from pyproj import Transformer
except Exception:
    Transformer = None

try:
    import shapely.geometry as geom
except Exception:
    geom = None

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


# ---------------------------
# Styling & Helpers
# ---------------------------
_styles_cache = None

def _styles():
    global _styles_cache
    if _styles_cache is not None:
        return _styles_cache
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='SectionHeading', fontSize=14, leading=18, spaceAfter=12,
        textColor=colors.HexColor("#0052cc"), fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        name='BodyTextDark', fontSize=10, leading=13,
        textColor=colors.HexColor("#1a1a1a"), spaceAfter=8
    ))
    styles.add(ParagraphStyle(
        name='TitleModern', fontSize=18, leading=22, alignment=1,
        textColor=colors.HexColor("#003087"), fontName='Helvetica-Bold',
        spaceAfter=20
    ))
    # New: wrapped body style for table cells to prevent overflow
    styles.add(ParagraphStyle(
        name='BodyWrap', parent=styles['BodyTextDark'], wordWrap='CJK'
    ))
    styles.add(ParagraphStyle(
        name='SmallGrey', fontSize=8, leading=10, textColor=colors.HexColor("#666666")
    ))
    _styles_cache = styles
    return styles


def _fmt(value: Any, fmt: str = ".2f") -> str:
    if value is None:
        return "N/A"
    try:
        return format(float(value), fmt)
    except Exception:
        return str(value)


def _maybe_add_image(elements: List[Any], path: Optional[str], title: str,
                     width_in: float = 6.5, height_in: float = 4.5):
    if path and os.path.exists(path):
        elements.append(Paragraph(title, _styles()['SectionHeading']))
        elements.append(Image(path, width=width_in * inch, height=height_in * inch))
        elements.append(Spacer(1, 12))


def _table(data: List[List[Any]], col_widths: Optional[List[float]] = None,
           heading_bg: str = "#e6f3ff", grid: bool = True, align: str = 'LEFT',
           repeat_heading: bool = False, compact: bool = True) -> Table:
    tbl = Table(data, colWidths=col_widths, hAlign=align, repeatRows=1 if repeat_heading else 0)
    style_cmds = [
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(heading_bg)),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5faff")]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('WORDWRAP', (0, 0), (-1, -1), 'CJK'),
    ]
    if grid:
        style_cmds += [
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#4d4d4d")),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#0052cc")),
        ]
    if compact:
        style_cmds += [
            ('LEFTPADDING', (0,0), (-1,-1), 4),
            ('RIGHTPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]
    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def _acceptance_badge(value: Optional[float], target: Optional[float]) -> str:
    if value is None or target is None:
        return "N/A"
    try:
        value_f = float(value)
        target_f = float(target)
        return "PASS" if value_f <= target_f else "CHECK"
    except Exception:
        return "N/A"


# ---------------------------
# Public API
# ---------------------------

def create_survey_summary_pdf(
    output_path: str,
    polygon_coords: List[Tuple[float, float]],
    gcp_points: List[Dict[str, Any]],
    veri_points: List[Dict[str, Any]],
    spacing: float,
    layout_mode: str,
    area_acres: float,
    elevation_data: Dict[str, Any],
    dsm_summary: Optional[str] = None,
    overlay_image_path: Optional[str] = None,
    # New: optional pre-rendered coverage heatmap path (if caller created one)
    coverage_heatmap_path: Optional[str] = None,
    flight_timings: Optional[List[float]] = None,
    pattern_score: float = 0.0,
    project_name: str = "UAS Survey",
    logo_path: Optional[str] = None,
    operator_name: Optional[str] = None,
    survey_location: Optional[str] = None,
    notes: Optional[str] = None,
    kmz_link: Optional[str] = None,
    # Legacy 3D image path (back-compat) or preferred model_3d_path
    report_path: Optional[str] = None,
    model_3d_path: Optional[str] = None,
    flight_params: Optional[Dict[str, Any]] = None,
    rejected_gcp_points: Optional[List[Dict[str, Any]]] = None,
    rejected_verification_points: Optional[List[Dict[str, Any]]] = None,
    rejected_veri_points: Optional[List[Dict[str, Any]]] = None,
    # New optional accuracy stats for QA/QC block
    accuracy_stats: Optional[Dict[str, Any]] = None,  # keys: rmse_v_m, rmse_h_m, mean_err_m, sd_m
    # Acceptance targets (defaults from your program requirements)
    target_vertical_rmse_m: float = 0.06096,  # 0.2 US ft @ 95% conf
    target_horizontal_rmse_m: Optional[float] = None,
    confidence_level: str = "95%",
    # Equipment / method metadata (optional, used in polished sections)
    equipment: Optional[Dict[str, Any]] = None,  # {uas, sensor, gnss_rover, base, corrections, firmware}
    data_lineage: Optional[List[str]] = None,    # bullet list of processing lineage
    version_tag: Optional[str] = None,           # e.g., "UAS Survey Tool v2.0 • Build 2025.08.09"
    include_standard_and_hybrid: bool = True,    # include both methodology options by default
) -> None:
    """
    Create a polished PDF survey summary report.

    Notes:
      - Accepts 'model_3d_path' (new) or 'report_path' (legacy) for a 3D figure.
      - Accepts both 'rejected_verification_points' and 'rejected_veri_points' (merged).
      - New optional sections: Executive Summary, QA/QC Acceptance, Dual Methodology, Sign-off.
    """
    # Harmonize alt-arg names
    if rejected_verification_points is None and rejected_veri_points:
        rejected_verification_points = rejected_veri_points

    three_d_image = model_3d_path or report_path

    tmp_files: List[str] = []

    try:
        # Document setup
        doc = SimpleDocTemplate(
            output_path, pagesize=letter,
            leftMargin=0.5 * inch, rightMargin=0.5 * inch,
            topMargin=0.75 * inch, bottomMargin=0.75 * inch
        )

        styles = _styles()
        elements: List[Any] = []

        # Logo
        if logo_path and os.path.exists(logo_path):
            elements.append(Image(logo_path, width=1.5 * inch, height=1.0 * inch))
            elements.append(Spacer(1, 10))

        # Title
        title_html = (
            f"<b>{project_name}</b><br/>"
            f"<i>{datetime.date.today():%B %d, %Y}</i>"
        )
        elements.append(Paragraph(title_html, styles['TitleModern']))
        elements.append(Spacer(1, 10))

        # Meta & Executive Summary
        kmz_td = Paragraph(f"<a href='{kmz_link}' color='blue'>{kmz_link}</a>", styles['BodyTextDark']) \
            if kmz_link else "N/A"
        meta_data = [
            ["Operator", operator_name or "N/A"],
            ["Location", survey_location or "Not specified"],
            ["Notes", notes or "—"],
            ["KMZ Source", kmz_td]
        ]
        elements.append(_table(meta_data, [1.5*inch, 5*inch]))
        elements.append(Spacer(1, 8))

        # Executive KPIs
        prof = elevation_data.get('profile', elevation_data.get('elevations', [])) or []
        min_elev = float(np.min(prof)) if prof else elevation_data.get('min', 0.0)
        max_elev = float(np.max(prof)) if prof else elevation_data.get('max', 0.0)
        mean_elev = float(np.mean(prof)) if prof else elevation_data.get('mean', 0.0)
        variance = float(np.var(prof)) if prof else 0.0
        flight_str = ", ".join(_fmt(t, ".1f") for t in (flight_timings or [])) or "N/A"

        kpis = [
            ["Survey Area (acres)", _fmt(area_acres, ".2f")],
            ["GCP Spacing (m)", _fmt(spacing, ".2f")],
            ["Elevation Range (m)", f"{_fmt(min_elev,'.1f')} – {_fmt(max_elev,'.1f')}"],
            ["Mean Elevation (m)", _fmt(mean_elev, ".1f")],
            ["Layout Mode", layout_mode.capitalize()],
            ["Flight Timings (min)", flight_str],
            ["GCPs / VCPs", f"{len(gcp_points)} / {len(veri_points)}"],
            ["Pattern Score", _fmt(pattern_score, ".2f")],
        ]
        elements.append(_table(kpis, [2*inch, 4.5*inch]))
        elements.append(Spacer(1, 12))

        # Compliance targets block
        elements.append(Paragraph("Quality Targets & Compliance", styles['SectionHeading']))
        targets = [
            ["Vertical RMSE Target (" + confidence_level + ")", f"{_fmt(target_vertical_rmse_m, '.3f')} m"],
            ["Horizontal RMSE Target (" + confidence_level + ")", _fmt(target_horizontal_rmse_m, '.3f') if target_horizontal_rmse_m is not None else "N/A"],
        ]
        # If accuracy stats provided, add PASS/CHECK
        if accuracy_stats:
            vr = accuracy_stats.get('rmse_v_m')
            hr = accuracy_stats.get('rmse_h_m')
            me = accuracy_stats.get('mean_err_m')
            sd = accuracy_stats.get('sd_m')
            targets += [
                ["Measured Vertical RMSE (m)", _fmt(vr, '.3f') + f"  [{_acceptance_badge(vr, target_vertical_rmse_m)}]"],
                ["Measured Horizontal RMSE (m)", _fmt(hr, '.3f') + (f"  [{_acceptance_badge(hr, target_horizontal_rmse_m)}]" if target_horizontal_rmse_m is not None else "")],
                ["Mean Error (m)", _fmt(me, '.3f')],
                ["Standard Deviation (m)", _fmt(sd, '.3f')],
            ]
        elements.append(_table(targets, [3.0*inch, 3.5*inch]))
        elements.append(Spacer(1, 12))

        # Terrain analysis summary (optional)
        if dsm_summary:
            elements.append(Paragraph("Terrain Analysis", styles['SectionHeading']))
            elements.append(Paragraph(dsm_summary.replace("\n","<br/>"), styles['BodyTextDark']))
            elements.append(Spacer(1, 10))

        # Overlay map
        _maybe_add_image(elements, overlay_image_path, "Survey Layout Map")

        # Optional Coverage heatmap (if caller passed a pre-rendered path)
        _maybe_add_image(elements, coverage_heatmap_path, "Coverage Heatmap")

        # Elevation profile chart
        if prof:
            tmp_chart = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            try:
                plt.figure(figsize=(6,2.5))
                plt.plot(prof, linewidth=1.5)
                plt.axhline(mean_elev, linestyle='--', label=f"Mean {_fmt(mean_elev,'.1f')} m")
                sd = float(np.sqrt(variance)) if variance > 0 else 0.0
                if sd > 0:
                    plt.axhline(mean_elev + sd, linestyle=':', label=f"+SD {_fmt(sd,'.1f')} m")
                    plt.axhline(mean_elev - sd, linestyle=':', label=f"-SD {_fmt(sd,'.1f')} m")
                plt.title("Elevation Profile", fontsize=12)
                plt.xlabel("Sample Index", fontsize=10)
                plt.ylabel("Elevation (m)", fontsize=10)
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                plt.savefig(tmp_chart.name, dpi=200, bbox_inches='tight')
            finally:
                plt.close()
            elements.append(Paragraph("Elevation Profile Chart", styles['SectionHeading']))
            elements.append(Image(tmp_chart.name, width=6*inch, height=2.5*inch))
            elements.append(Spacer(1,12))
            tmp_files.append(tmp_chart.name)

        # 3D DSM/DEM rendition (optional quick viz)
        if prof and polygon_coords and len(prof) >= 4 and Transformer is not None and geom is not None:
            tmp_dsm = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            try:
                # Pick UTM from centroid for a simple surface visualization
                lons = [p[0] for p in polygon_coords]
                lats = [p[1] for p in polygon_coords]
                clon = sum(lons)/len(lons)
                clat = sum(lats)/len(lats)
                zone = int((clon + 180) // 6) + 1
                epsg = 32600 + zone if clat >= 0 else 32700 + zone
                transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
                utm = [transformer.transform(lon, lat) for lon, lat in polygon_coords]
                poly = geom.Polygon(utm)
                minx, miny, maxx, maxy = poly.bounds
                n = max(2, int(np.sqrt(len(prof))))
                x = np.linspace(minx, maxx, n)
                y = np.linspace(miny, maxy, n)
                X, Y = np.meshgrid(x, y)
                if len(prof) >= n*n:
                    Z = np.reshape(prof[:n*n], (n, n))
                    fig = plt.figure(figsize=(6.5,4.5))
                    ax3 = fig.add_subplot(111, projection='3d')
                    ax3.plot_surface(X, Y, Z, cmap='terrain')
                    ax3.set_xlabel('Easting (m)')
                    ax3.set_ylabel('Northing (m)')
                    ax3.set_zlabel('Elevation (m)')
                    plt.title('DSM/DEM Rendition')
                    plt.tight_layout()
                    plt.savefig(tmp_dsm.name, dpi=200, bbox_inches='tight')
                    plt.close()
                    elements.append(Paragraph("DSM/DEM Rendition", styles['SectionHeading']))
                    elements.append(Image(tmp_dsm.name, width=6.5*inch, height=4.5*inch))
                    elements.append(Spacer(1,12))
            finally:
                tmp_files.append(tmp_dsm.name)

        # Optional pre-rendered 3D model image
        _maybe_add_image(elements, three_d_image, "3D Terrain Model")

        # Rejected Points Summary
        if (rejected_gcp_points and len(rejected_gcp_points) > 0) or (rejected_verification_points and len(rejected_verification_points) > 0):
            elements.append(Paragraph("Rejected Points Summary", styles['SectionHeading']))
            data = [["ID", "Type", "Lat", "Lon", "Reason"]]
            for i, pt in enumerate(rejected_gcp_points or []):
                data.append([
                    f"RGCP_{i+1}", "GCP",
                    _fmt(pt.get('lat'), '.6f'), _fmt(pt.get('lon'), '.6f'),
                    pt.get('reason', 'N/A')
                ])
            for i, pt in enumerate(rejected_verification_points or []):
                data.append([
                    f"RVER_{i+1}", "Verification",
                    _fmt(pt.get('lat'), '.6f'), _fmt(pt.get('lon'), '.6f'),
                    pt.get('reason', 'N/A')
                ])
            # Page chunking
            rows_per_page = 16
            for i in range(0, len(data), rows_per_page):
                chunk = data[i:i + rows_per_page]
                elements.append(_table(chunk, [0.9*inch, 1.1*inch, 1.2*inch, 1.2*inch, 2.1*inch], align='CENTER'))
                if i + rows_per_page < len(data):
                    elements.append(PageBreak())
            elements.append(Spacer(1, 10))

        # Dual Methodology Section (Hybrid + Standard)
        if include_standard_and_hybrid:
            _add_dual_methodology(elements)

        # Equipment & Data Lineage (optional polish)
        if equipment or data_lineage:
            elements.append(Paragraph("Equipment & Data Lineage", styles['SectionHeading']))
            eq_rows = [["Item", "Details"]]
            if equipment:
                for k in ["uas", "sensor", "gnss_rover", "base", "corrections", "firmware"]:
                    if equipment.get(k):
                        label = {
                            "uas": "UAS Platform",
                            "sensor": "Sensor / Payload",
                            "gnss_rover": "GNSS Rover",
                            "base": "Base / Network",
                            "corrections": "Corrections",
                            "firmware": "Firmware"
                        }[k]
                        eq_rows.append([label, str(equipment.get(k))])
            if data_lineage:
                eq_rows.append(["Processing Lineage", " → ".join(data_lineage)])
            elements.append(_table(eq_rows, [1.8*inch, 4.7*inch]))
            elements.append(Spacer(1, 10))

        # Sign-off block
        elements.append(Paragraph("Certification & Sign-Off", styles['SectionHeading']))
        sign_rows = [
            ["Prepared By", operator_name or "N/A"],
            ["Date", f"{datetime.date.today():%Y-%m-%d}"],
            ["Signature", "______________________________"],
        ]
        elements.append(_table(sign_rows, [1.6*inch, 4.9*inch]))
        elements.append(Spacer(1, 6))
        if version_tag:
            elements.append(Paragraph(version_tag, styles['SmallGrey']))

        # Build PDF
        doc.build(elements)

    finally:
        for p in tmp_files:
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


# ---------------------------
# Section Builders
# ---------------------------

def _add_dual_methodology(elements: List[Any]) -> None:
    styles = _styles()
    elements.append(Paragraph("Field Survey Methodologies", styles['SectionHeading']))
    intro = (
        "This project supports two vetted procedures for collecting Ground Control Points (GCPs) and "
        "Verification Check Points (VCPs). Survey leads may choose either approach based on site conditions, "
        "crew readiness, and schedule. Both methods are compatible with the tool’s QA/QC pipeline."
    )
    elements.append(Paragraph(intro, styles['BodyTextDark']))
    elements.append(Spacer(1, 8))

    # Build wrapped Paragraph cells to avoid overflow; total width = 7.5 in (letter width minus 0.5in margins)
    def P(text: str) -> Paragraph:
        return Paragraph(text, styles['BodyWrap'])

    header = [P("Method"), P("GCP Procedure"), P("VCP Procedure"), P("Notes")]
    row_hybrid = [
        P("HYBRID (Rotational)"),
        P("3 × 60s GNSS shots per GCP with 120° rod rotations (per shot)."),
        P("3 × 10s GNSS shots per VCP with 120° rod rotations (per shot)."),
        P("Optimized for robustness; rotational averaging reduces orientation bias."),
    ]
    row_standard = [
        P("STANDARD (Static)"),
        P("Single 180s static GNSS observation per GCP. No rod rotation."),
        P("Single 30s static GNSS observation per VCP. No rod rotation."),
        P("Faster to train and execute; ideal for straightforward sites and clear sky."),
    ]

    comp = [header, row_hybrid, row_standard]
    # Column widths sum to 7.5 in (page width inside margins)
    col_widths = [1.3*inch, 2.4*inch, 2.4*inch, 1.4*inch]
    elements.append(_table(comp, col_widths, heading_bg="#e6f3ff", grid=True, align='LEFT', repeat_heading=True, compact=True))
    elements.append(Spacer(1, 10))

    # Hybrid details
    elements.append(Paragraph("Hybrid Method — Detailed Steps", styles['BodyTextDark']))
    steps_h = [
        "Pre-mission: plan flight, select ~3+ representative GCP and VCP locations (open, vegetated, transitions).",
        "Each GCP: collect three 60-second shots; for each shot, rotate the rod ~120° between observations.",
        "Each VCP: collect three 10-second shots with ~120° rotations between observations.",
        "Post-process to compute averaged position per point; verify deltas vs. LiDAR/DEM; compute RMSE/ME/SD.",
    ]
    elements.append(_bullet_list(steps_h))
    elements.append(Spacer(1, 6))

    # Standard details
    elements.append(Paragraph("Standard Method — Detailed Steps", styles['BodyTextDark']))
    steps_s = [
        "Each GCP: single continuous 180-second static GNSS observation (no rod rotation).",
        "Each VCP: single continuous 30-second static GNSS observation (no rod rotation).",
        "Validate signal quality (PDOP, residuals). Mark positions; proceed with post-processing and QA/QC.",
    ]
    elements.append(_bullet_list(steps_s))
    elements.append(Spacer(1, 6))

    # Best practices
    elements.append(Paragraph("Best Practices & Selection Guidance", styles['BodyTextDark']))
    tips = [
        "Use HYBRID when multipath or wind may influence rod orientation or fast repeatability is desired.",
        "Use STANDARD for simpler terrain with stable sky view to minimize setup time.",
        "Target vertical RMSE ≤ 0.06096 m (0.2 US ft) at 95% confidence; add VCPs if acceptance is at risk.",
        "Capture site photos at each control for as-built traceability (optional).",
    ]
    elements.append(_bullet_list(tips))
    elements.append(Spacer(1, 10))


def _bullet_list(items: List[str]) -> Table:
    # Render a clean two-column table with bullets for consistent alignment; avoid special glyphs
    rows = []
    for it in items:
        rows.append(["•", Paragraph(it, _styles()['BodyWrap'])])
    tbl = Table(rows, colWidths=[0.2*inch, 6.3*inch], hAlign='LEFT')
    tbl.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (0,0), (0,-1), 'RIGHT'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    return tbl
