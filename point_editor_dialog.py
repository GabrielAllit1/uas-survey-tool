# point_editor_dialog.py
from __future__ import annotations

import csv
import logging
import os
from typing import Optional, Tuple, List

import numpy as np
from shapely.geometry import Point, Polygon as ShpPolygon
from pyproj import Transformer, CRS

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget, QListWidgetItem,
    QDialogButtonBox, QMessageBox, QSlider
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from unified_mathtoolbox import MathToolBox
from elevation_data import get_srtm_elevation_bulk
from flight_parameters_calculator import calculate_flight_parameters

# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
log_dir = os.path.expanduser("~/UAS_Survey_Tool_Logs")
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'uas_survey_tool.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
_TOOLBOX_SINGLETON: Optional[MathToolBox] = None

def get_toolbox() -> MathToolBox:
    global _TOOLBOX_SINGLETON
    if _TOOLBOX_SINGLETON is None:
        try:
            _TOOLBOX_SINGLETON = MathToolBox(use_cuda=False)
            logger.info("MathToolBox initialized on CPU")
        except Exception:
            # Fallback to a minimal stub if import fails for any reason
            class _Stub:
                def fibonacci(self, n: int) -> int:
                    a, b = 0, 1
                    for _ in range(n):
                        a, b = b, a + b
                    return a
                def unified_detection(self, x, k: int = 32, bias: float = 0.0) -> float:
                    arr = np.array([v for v in (x if isinstance(x, (list, tuple, np.ndarray)) else [x]) if v is not None], dtype=float)
                    if arr.size < 3:
                        return 0.5
                    s_all = np.std(arr) + 1e-9
                    s_d = np.std(np.diff(arr)) if arr.size >= 3 else 0.0
                    return float(np.clip(s_d / s_all, 0, 1))
            _TOOLBOX_SINGLETON = _Stub()  # type: ignore
    return _TOOLBOX_SINGLETON  # type: ignore

def _multi_dataset_fetch(lat: float, lon: float, retries: int = 2) -> Optional[float]:
    """
    Attempt multiple OpenTopoData datasets to avoid default fallbacks.
    Tries: ned10m -> srtm90m -> aster30m (if available via your OpenTopo proxy).
    Returns a positive float on success, or None on failure.
    """
    datasets = ["ned10m", "srtm90m", "aster30m"]
    for ds in datasets:
        try:
            elevs, _ = get_srtm_elevation_bulk([(lat, lon)], retry=max(1, retries), dataset=ds)
            v = elevs[0] if elevs else None
            if v is not None and isinstance(v, (int, float)) and float(v) > 0:
                return float(v)
        except Exception as e:
            logger.debug(f"Dataset {ds} fetch failed at ({lat:.6f},{lon:.6f}): {e}")
    return None

def _safe_fetch_elevation(lat: float, lon: float, default_elev: float) -> float:
    """Try multiple datasets with small retries; fall back to default on any issue."""
    try:
        v = _multi_dataset_fetch(lat, lon, retries=2)
        if v is not None and v > 0:
            return v
    except Exception as e:
        logger.warning("Elevation fetch failed at (%.6f, %.6f): %s", lat, lon, e)
    return float(default_elev)

# --------------------------------------------------------------------------------------
# Background thread to regenerate points at a different density (kept from prior behavior)
# --------------------------------------------------------------------------------------
class DensityAdjustmentThread(QThread):
    finished = pyqtSignal(list, list)
    error = pyqtSignal(str)

    def __init__(
        self,
        main_logic,
        gcp_points: List[dict],
        verification_points: List[dict],
        density_factor: float,
        area_m2: float,
        user_spacing: float,
        layout_mode: str,
        sequence_type: str,
        dem_path: Optional[str],
        dsm_path: Optional[str],
        min_distance_factor: float,
        accuracy_mm: float,
        weights: List[float],
    ):
        super().__init__()
        self.main_logic = main_logic
        self.gcp_points = gcp_points
        self.verification_points = verification_points
        self.density_factor = density_factor
        self.area_m2 = area_m2
        self.user_spacing = user_spacing
        self.layout_mode = layout_mode
        self.sequence_type = sequence_type
        self.dem_path = dem_path
        self.dsm_path = dsm_path
        self.min_distance_factor = min_distance_factor
        self.accuracy_mm = accuracy_mm
        self.weights = weights

    def run(self):
        try:
            from gcp_generator import generate_gcp
            # Use toolbox sequences (if available)
            fib_seq = [get_toolbox().fibonacci(i) for i in range(1001)]
            kap_seq = [0.5] * 1000  # placeholder scalar weights

            gcp_points, verification_points, _rej_g, _rej_v = generate_gcp(
                geometry_wgs84=self.main_logic.polygon,
                spacing=self.user_spacing,
                layout_mode=self.layout_mode,
                sequence_type=self.sequence_type,
                modulus=self.main_logic.calculate_modulus(self.main_logic.polygon_area_acres, len(self.gcp_points) or 1),
                min_distance_factor=self.min_distance_factor,
                dem_path=self.dem_path,
                dsm_path=self.dsm_path,
                fib_sequence=fib_seq,
                kap_sequence=kap_seq,
                main_logic=self.main_logic,
                density_factor=self.density_factor,
                gcp_density=self.user_spacing,
                manual_override=False  # this is auto-regeneration
            )
            self.finished.emit(gcp_points, verification_points)
        except Exception as e:
            logger.error("Density adjustment failed: %s", e)
            self.error.emit(str(e))

# --------------------------------------------------------------------------------------
# Dialog
# --------------------------------------------------------------------------------------
class PointEditorDialog(QDialog):
    """
    Allows interactive review/add/remove of GCP and VCP points within the AOI.
    Save: writes to cache CSV and updates main_logic + parent state, then closes.
    Save & Export: does Save, then calls parent().on_export_clicked() to regenerate all deliverables.
    """
    def __init__(self, main_logic, gcp_points, verification_points, area_m2, parent=None):
        super().__init__(parent)
        self.main_logic = main_logic
        self.gcp_points: List[dict] = list(gcp_points or [])
        self.verification_points: List[dict] = list(verification_points or [])
        self.rejected_gcp_points: List[dict] = []
        self.rejected_verification_points: List[dict] = []
        self.area_m2 = float(area_m2 or 0.0)
        self.density_factor = 1.0
        self.desired_accuracy_mm = 60.96  # 0.2 ft at 95% ~ 60.96 mm

        # cache default csv path
        self.cache_dir = os.path.expanduser("~/.uas_survey_tool_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.default_csv_path = os.path.join(self.cache_dir, "points.csv")

        self._init_ui()
        self._populate_list(self.gcp_list, self.gcp_points, 'GCP')
        self._populate_list(self.ver_list, self.verification_points, 'VCP')
        self._plot_points()

    # ---------------- UI setup ----------------
    def _init_ui(self):
        self.setWindowTitle("Point Editor")
        layout = QVBoxLayout()

        # Lists for GCP and VCP
        lists_layout = QHBoxLayout()
        self.gcp_list = QListWidget()
        self.ver_list = QListWidget()
        lists_layout.addWidget(QLabel("GCPs"))
        lists_layout.addWidget(self.gcp_list)
        lists_layout.addWidget(QLabel("VCPs"))
        lists_layout.addWidget(self.ver_list)
        layout.addLayout(lists_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_add_gcp = QPushButton("Add GCP")
        self.btn_add_vcp = QPushButton("Add VCP")
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_regenerate = QPushButton("Regenerate Points")

        # NEW: Save buttons
        self.btn_save_only = QPushButton("Save")
        self.btn_save_export = QPushButton("Save & Export")

        btn_layout.addWidget(self.btn_add_gcp)
        btn_layout.addWidget(self.btn_add_vcp)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_regenerate)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.btn_save_only)
        btn_layout.addWidget(self.btn_save_export)
        layout.addLayout(btn_layout)

        # Density Slider
        density_layout = QHBoxLayout()
        density_layout.addWidget(QLabel("Density Adjustment"))
        self.density_slider = QSlider(Qt.Orientation.Horizontal)
        self.density_slider.setMinimum(1)
        self.density_slider.setMaximum(100)
        self.density_slider.setValue(50)
        self.density_slider.valueChanged.connect(self._on_density_change)
        density_layout.addWidget(self.density_slider)
        layout.addLayout(density_layout)

        # Plot
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Dialog buttons (Close only; save is handled by our explicit buttons)
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

        # Connect signals
        self.btn_add_gcp.clicked.connect(lambda: self._add_point('GCP'))
        self.btn_add_vcp.clicked.connect(lambda: self._add_point('VCP'))
        self.btn_remove.clicked.connect(self._remove_point)
        self.btn_regenerate.clicked.connect(self._regenerate_points)
        self.btn_save_only.clicked.connect(self._on_save_clicked)
        self.btn_save_export.clicked.connect(self._on_save_and_export_clicked)

    # ---------------- UI interactions ----------------
    def _on_density_change(self):
        self.density_factor = self.density_slider.value() / 50.0

    def _populate_list(self, widget: QListWidget, points: List[dict], kind: str):
        widget.clear()
        for pt in points:
            name = pt.get('name') or f"{kind}_{len(points)}"
            e = pt.get('easting', 0.0)
            n = pt.get('northing', 0.0)
            item = QListWidgetItem(f"{name}  ({e:.2f}, {n:.2f})")
            widget.addItem(item)

    def _plot_points(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # AOI polygon (projected)
        try:
            poly: Optional[ShpPolygon] = getattr(self.main_logic, "polygon_proj", None)
            if poly is None:
                # build projected polygon once if needed
                if self.main_logic.polygon is not None:
                    crs = getattr(self.main_logic, "crs", None) or CRS.from_epsg(4326)
                    tf = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
                    poly = ShpPolygon([tf.transform(lon, lat) for lon, lat in getattr(self.main_logic, "polygon_coords", [])])
                else:
                    poly = None
            if poly and not poly.is_empty:
                x, y = poly.exterior.xy
                ax.plot(x, y, "k-", linewidth=1.25, label="Survey Area")
        except Exception as e:
            logger.warning(f"Polygon plot failed: {e}")

        # Points
        def scat(pts, label, marker, color):
            if not pts:
                return
            xs = [p.get('easting', 0.0) for p in pts]
            ys = [p.get('northing', 0.0) for p in pts]
            ax.scatter(xs, ys, marker=marker, s=24, alpha=0.95, label=label, c=color)

        scat(self.gcp_points, "GCP", "o", "tab:blue")
        scat(self.verification_points, "VCP", "x", "tab:red")

        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)
        ax.legend(loc="best")
        self.canvas.draw_idle()

    def _add_point(self, kind: str):
        """
        Put the canvas into a one-click add mode for the given point type.
        """
        QMessageBox.information(self, "Add Point",
                                f"Click on the map to add a new {kind}.")
        cid = {}

        def onclick(ev):
            if ev.inaxes is None:
                return
            e, n = float(ev.xdata), float(ev.ydata)
            # to lon/lat
            tf = Transformer.from_crs(self.main_logic.crs, "EPSG:4326", always_xy=True)
            lon, lat = tf.transform(e, n)
            elev = _safe_fetch_elevation(lat, lon, default_elev=self.main_logic.centroid_elevation)
            name_prefix = "GCP" if kind == "GCP" else "VCP"
            seq = len(self.gcp_points) + 1 if kind == "GCP" else len(self.verification_points) + 1
            pd = {
                "name": f"{name_prefix}_{seq}",
                "type": kind,
                "easting": e,
                "northing": n,
                "lon": lon,
                "lat": lat,
                "elevation": float(elev)
            }
            if kind == "GCP":
                self.gcp_points.append(pd)
                self._populate_list(self.gcp_list, self.gcp_points, "GCP")
            else:
                self.verification_points.append(pd)
                self._populate_list(self.ver_list, self.verification_points, "VCP")
            self._plot_points()
            # disconnect after single add
            self.canvas.mpl_disconnect(cid["id"])

        cid["id"] = self.canvas.mpl_connect("button_press_event", onclick)

    def _remove_point(self):
        """
        Remove the currently selected items from the lists.
        """
        # Remove GCPs
        for idx in sorted({i.row() for i in self.gcp_list.selectedIndexes()}, reverse=True):
            try:
                self.gcp_points.pop(idx)
                self.gcp_list.takeItem(idx)
            except Exception:
                pass
        # Remove VCPs
        for idx in sorted({i.row() for i in self.ver_list.selectedIndexes()}, reverse=True):
            try:
                self.verification_points.pop(idx)
                self.ver_list.takeItem(idx)
            except Exception:
                pass
        self._plot_points()

    def _regenerate_points(self):
        """
        Optional: re-run generator at adjusted density (non-blocking).
        """
        try:
            spacing = float(self.main_logic.calculate_recommended_spacing_from_payload())
            th = DensityAdjustmentThread(
                main_logic=self.main_logic,
                gcp_points=self.gcp_points,
                verification_points=self.verification_points,
                density_factor=self.density_factor,
                area_m2=self.area_m2,
                user_spacing=spacing,
                layout_mode=self.main_logic.layout_mode,
                sequence_type=self.main_logic.sequence_type,
                dem_path=self.main_logic.dem_path,
                dsm_path=self.main_logic.dsm_path,
                min_distance_factor=0.65,
                accuracy_mm=self.desired_accuracy_mm,
                weights=getattr(self.main_logic, "weights", [0.25, 0.25, 0.25, 0.25]),
            )
            def _ok(g, v):
                self.gcp_points = list(g or [])
                self.verification_points = list(v or [])
                self._populate_list(self.gcp_list, self.gcp_points, "GCP")
                self._populate_list(self.ver_list, self.verification_points, "VCP")
                self._plot_points()
            th.finished.connect(_ok)
            th.error.connect(lambda m: QMessageBox.warning(self, "Regenerate Points", m))
            th.start()
        except Exception as e:
            QMessageBox.warning(self, "Regenerate Points", f"Failed: {e}")

    # ---------------- Save logic ----------------
    def _write_points_to_cache_csv(self):
        """
        Save all current points to ~/.uas_survey_tool_cache/points.csv
        Columns: Name,Type,Easting,Northing,Longitude,Latitude,Elevation
        """
        rows = []
        for pt in (self.gcp_points + self.verification_points):
            rows.append([
                pt.get("name", ""),
                pt.get("type", "GCP" if "GCP" in pt.get("name","") else "VCP"),
                float(pt.get("easting", 0.0)),
                float(pt.get("northing", 0.0)),
                float(pt.get("lon", 0.0)),
                float(pt.get("lat", 0.0)),
                float(pt.get("elevation", self.main_logic.centroid_elevation)),
            ])
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.default_csv_path, "w", newline="", encoding="utf-8") as fp:
            w = csv.writer(fp)
            w.writerow(["Name", "Type", "Easting", "Northing", "Longitude", "Latitude", "Elevation"])
            w.writerows(rows)
        logger.info("Saved points to cache CSV: %s", self.default_csv_path)

    def _apply_points_to_main_logic(self):
        """
        Update main_logic and (if present) parent's cached copies.
        """
        self.main_logic.gcp_points = list(self.gcp_points)
        self.main_logic.verification_points = list(self.verification_points)
        self.main_logic.rejected_gcp_points = list(self.rejected_gcp_points)
        self.main_logic.rejected_verification_points = list(self.rejected_verification_points)

        # If parent window keeps its own copies, keep them in sync
        try:
            parent = self.parent()
            if parent is not None:
                parent.gcp_points = list(self.gcp_points)
                parent.verification_points = list(self.verification_points)
                parent.rejected_gcp_points = list(self.rejected_gcp_points)
                parent.rejected_verification_points = list(self.rejected_verification_points)
        except Exception:
            pass

    def _on_save_clicked(self):
        """
        Save (to cache CSV + update main_logic) and close as 'accepted' (not cancelled).
        """
        try:
            # Ensure all points have good elevations (recheck any missing/new)
            for pt in (self.gcp_points + self.verification_points):
                lat = float(pt.get("lat", 0.0))
                lon = float(pt.get("lon", 0.0))
                elev = pt.get("elevation", None)
                if elev is None or not isinstance(elev, (int, float)) or float(elev) <= 0:
                    pt["elevation"] = _safe_fetch_elevation(lat, lon, self.main_logic.centroid_elevation)

            self._write_points_to_cache_csv()
            self._apply_points_to_main_logic()
            QMessageBox.information(self, "Saved", "Point edits saved.")
            self.accept()  # close dialog as saved
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))

    def _on_save_and_export_clicked(self):
        """
        Save (as above), then trigger parent export (CSV/KMZ/PDF/PNG updated with new points/elevations).
        """
        try:
            self._on_save_clicked()  # does elevation, cache csv, main_logic update, and accept()
            # After accept, dialog is closing; still try to call export on parent
            parent = self.parent()
            if parent is not None and hasattr(parent, "on_export_clicked"):
                try:
                    parent.on_export_clicked()
                except Exception as e:
                    logger.warning(f"Export trigger failed: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Save & Export Failed", str(e))
