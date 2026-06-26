import sys
import os
import logging
import time
from datetime import datetime
import csv
import zipfile
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QComboBox, QLineEdit, QLabel, QFileDialog, QGroupBox,
    QTextEdit, QCheckBox, QSizePolicy, QStatusBar, QMessageBox, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QIcon, QDoubleValidator
try:
    import pycuda
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False
from main_logic import MainLogic
from overlay_diagnostics import generate_overlay_diagnostics
from overlay_map_generator import generate_overlay_map, auto_detect_utm_crs
from point_editor_dialog import PointEditorDialog
from pdf_report import create_survey_summary_pdf
from dsm_manager import get_dsm_elevation_profile, suggest_layout_based_on_dsm
from elevation_data import get_srtm_elevation_bulk
from api_key_manager import APIKeyManager

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

class TextEditHandler(logging.Handler):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record):
        msg = self.format(record)
        def append_log():
            self.text_edit.append(msg)
            self.text_edit.ensureCursorVisible()
        if QThread.currentThread() is QApplication.instance().thread():
            append_log()
        else:
            QTimer.singleShot(0, append_log)

class PlanGenerationThread(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, window):
        super().__init__()
        self.window = window

    def run(self):
        try:
            self.status.emit("Generating GCP plan...")
            user_spacing = None
            if self.window.spacing_input.text():
                try:
                    user_spacing = float(self.window.spacing_input.text())
                    if not 1.0 <= user_spacing <= 1000.0:
                        raise ValueError("Spacing must be between 1.0 and 1000.0 meters")
                except ValueError as e:
                    self.error.emit(str(e))
                    return
            weights = None
            if self.window.weights_input.text():
                try:
                    weights = [float(w) for w in self.window.weights_input.text().split(',')]
                    if len(weights) != 4:
                        raise ValueError("Weights must be 4 comma-separated floats")
                    if any(w < 0 for w in weights):
                        raise ValueError("Weights must be non-negative")
                except ValueError as e:
                    self.error.emit(str(e))
                    return
            try:
                accuracy = float(self.window.accuracy_input.text())
                altitude = float(self.window.altitude_input.text())
                fov = float(self.window.fov_input.text())
                prr = float(self.window.prr_input.text()) * 1000  # Convert kHz to Hz
                self.window.main_logic.compute_flight_parameters(accuracy, altitude, fov, prr)
            except ValueError as e:
                self.error.emit(f"Invalid flight parameters: {e}")
                return
            self.window.main_logic.set_survey_method(self.window.survey_method_combo.currentText())
            dem_path = self.window.dem_input.text() if self.window.dem_input.text() else None
            dsm_path = self.window.dsm_input.text() if self.window.dsm_input.text() else None
            auto_download_dem = self.window.dem_check.isChecked()
            auto_download_dsm = self.window.dsm_check.isChecked()
            if auto_download_dem and not dem_path and self.window.main_logic.last_kmz_path and os.path.exists(self.window.main_logic.last_kmz_path):
                try:
                    dem_result = self.window.main_logic.download_dem(self.window.main_logic.last_kmz_path)
                    if isinstance(dem_result, tuple):
                        dem_path, centroid_elevation = dem_result
                        self.window.main_logic.centroid_elevation = centroid_elevation
                        self.status.emit(f"Using centroid elevation {centroid_elevation:.2f} m (no valid DEM)")
                    else:
                        dem_path = dem_result
                        self.status.emit(f"Downloaded DEM to {dem_path}")
                    self.window.dem_input.setText(dem_path if dem_path else "")
                except Exception as e:
                    self.window.logger.warning(f"Auto DEM download failed: {e}")
                    self.status.emit(f"Auto DEM download failed: {e}")
                    dem_path = None
                    self.window.main_logic.centroid_elevation = 291.0
            if auto_download_dsm and not dsm_path and self.window.main_logic.last_kmz_path and os.path.exists(self.window.main_logic.last_kmz_path):
                try:
                    dsm_result = self.window.main_logic.download_dsm(self.window.main_logic.last_kmz_path)
                    if isinstance(dsm_result, tuple):
                        dsm_path, centroid_elevation = dsm_result
                        self.window.main_logic.centroid_elevation = centroid_elevation
                        self.status.emit(f"Using centroid elevation {centroid_elevation:.2f} m (no valid DSM)")
                    else:
                        dsm_path = dsm_result
                        self.status.emit(f"Downloaded DSM to {dsm_path}")
                    self.window.dsm_input.setText(dsm_path if dsm_path else "")
                except Exception as e:
                    self.window.logger.warning(f"Auto DSM download failed: {e}")
                    self.status.emit(f"Auto DSM download failed: {e}")
                    dsm_path = None
            if not dem_path and not dsm_path and (auto_download_dem or auto_download_dsm):
                self.window.logger.warning("No valid DSM/DEM path after auto-download attempts")
                self.status.emit(f"Warning: No valid DSM/DEM path; using centroid elevation {self.window.main_logic.centroid_elevation:.2f} m")
            gcp_points, verification_points, elevation_data = self.window.main_logic.generate_gcp_plan(
                auto_download_dem=auto_download_dem,
                auto_download_dsm=auto_download_dsm,
                user_spacing=user_spacing,
                layout_mode=self.window.layout_combo.currentText().lower(),
                sequence_type=self.window.sequence_combo.currentText().lower(),
                weights=weights,
                dem_path=dem_path,
                dsm_path=dsm_path
            )
            result = {
                "gcp_points": gcp_points,
                "verification_points": verification_points,
                "rejected_gcp_points": getattr(self.window.main_logic, 'rejected_gcp_points', []),
                "rejected_verification_points": getattr(self.window.main_logic, 'rejected_verification_points', []),
                "elevation_data": elevation_data
            }
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"GCP plan generation failed: {str(e)}")
            self.window.logger.error(f"PlanGenerationThread failed: {str(e)}")

class ExportDeliverablesThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    status = pyqtSignal(str)
    overwrite_check = pyqtSignal(str, str, str, str)

    def __init__(self, window, output_dir):
        super().__init__()
        self.window = window
        self.output_dir = output_dir

    def run(self):
        try:
            self.status.emit("Exporting deliverables...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            project_name = self.window.area_name or "survey"
            csv_path = os.path.join(self.output_dir, f"{project_name}_{timestamp}_points.csv")
            kml_path = os.path.join(self.output_dir, f"{project_name}_{timestamp}.kml")
            report_path = os.path.join(self.output_dir, f"{project_name}_{timestamp}_report.pdf")
            overlay_path = os.path.join(self.output_dir, f"{project_name}_{timestamp}_overlay.png")
            if os.path.exists(csv_path) or os.path.exists(kml_path) or os.path.exists(report_path) or os.path.exists(overlay_path):
                self.overwrite_check.emit(csv_path, kml_path, report_path, overlay_path)
                return
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Name', 'Easting', 'Northing', 'Elevation'])
                for pt in self.window.gcp_points + self.window.verification_points:
                    writer.writerow([pt['name'], pt['easting'], pt['northing'], pt.get('elevation', self.window.main_logic.centroid_elevation)])
                self.window.logger.info(f"Exported Trimble-compatible CSV to {csv_path}")
            from simplekml import Kml
            kml = Kml()
            for pt in self.window.gcp_points:
                p = kml.newpoint(name=pt['name'], coords=[(pt['lon'], pt['lat'], pt.get('elevation', self.window.main_logic.centroid_elevation))])
                p.description = f"GCP: Easting={pt['easting']:.2f}, Northing={pt['northing']:.2f}, Elevation={pt.get('elevation', self.window.main_logic.centroid_elevation):.2f}"
            for pt in self.window.verification_points:
                p = kml.newpoint(name=pt['name'], coords=[(pt['lon'], pt['lat'], pt.get('elevation', self.window.main_logic.centroid_elevation))])
                p.description = f"VCP: Easting={pt['easting']:.2f}, Northing={pt['northing']:.2f}, Elevation={pt.get('elevation', self.window.main_logic.centroid_elevation):.2f}"
            kml.save(kml_path)
            self.window.logger.info(f"Exported KML to {kml_path}")
            dsm_or_dem_path = getattr(self.window.main_logic, "dsm_path", None) or getattr(self.window.main_logic, "dem_path", None)
            if not dsm_or_dem_path and (self.window.dem_check.isChecked() or self.window.dsm_check.isChecked()):
                if self.window.dem_check.isChecked() and self.window.main_logic.last_kmz_path and os.path.exists(self.window.main_logic.last_kmz_path):
                    try:
                        dem_result = self.window.main_logic.download_dem(self.window.main_logic.last_kmz_path)
                        if isinstance(dem_result, tuple):
                            dsm_or_dem_path, centroid_elevation = dem_result
                            self.window.main_logic.centroid_elevation = centroid_elevation
                            self.status.emit(f"Using centroid elevation {centroid_elevation:.2f} m (no valid DEM)")
                        else:
                            dsm_or_dem_path = dem_result
                            self.status.emit(f"Downloaded DEM to {dsm_or_dem_path}")
                        self.window.dem_input.setText(dsm_or_dem_path if dsm_or_dem_path else "")
                        self.window.main_logic.dem_path = dsm_or_dem_path
                    except Exception as e:
                        self.window.logger.warning(f"Auto DEM download failed: {e}")
                        self.status.emit(f"Auto DEM download failed: {e}")
                        dsm_or_dem_path = None
                elif self.window.dsm_check.isChecked() and self.window.main_logic.last_kmz_path and os.path.exists(self.window.main_logic.last_kmz_path):
                    try:
                        dsm_result = self.window.main_logic.download_dsm(self.window.main_logic.last_kmz_path)
                        if isinstance(dsm_result, tuple):
                            dsm_or_dem_path, centroid_elevation = dsm_result
                            self.window.main_logic.centroid_elevation = centroid_elevation
                            self.status.emit(f"Using centroid elevation {centroid_elevation:.2f} m (no valid DSM)")
                        else:
                            dsm_or_dem_path = dsm_result
                            self.status.emit(f"Downloaded DSM to {dsm_or_dem_path}")
                        self.window.dsm_input.setText(dsm_or_dem_path if dsm_or_dem_path else "")
                        self.window.main_logic.dsm_path = dsm_or_dem_path
                    except Exception as e:
                        self.window.logger.warning(f"Auto DSM download failed: {e}")
                        self.status.emit(f"Auto DSM download failed: {e}")
                        dsm_or_dem_path = None
            overlay_success = False
            crs = getattr(self.window.main_logic, 'crs', None)
            if not crs and self.window.polygon_coords and len(self.window.polygon_coords) > 0:
                centroid_lon = sum(lon for lon, _ in self.window.polygon_coords) / len(self.window.polygon_coords)
                centroid_lat = sum(lat for _, lat in self.window.polygon_coords) / len(self.window.polygon_coords)
                crs = auto_detect_utm_crs(centroid_lon, centroid_lat)
                self.window.main_logic.crs = crs
                self.window.logger.debug(f"Set CRS from centroid: {crs}")
            elif not crs:
                crs = 'EPSG:32614'
                self.window.main_logic.crs = crs
                self.window.logger.warning("No polygon coordinates available; using default CRS: EPSG:32614")
            if dsm_or_dem_path and os.path.exists(dsm_or_dem_path):
                try:
                    detection_score = self.window.elevation_data.get('pattern_score', 0.0)
                    self.window.diagnostic_report = generate_overlay_diagnostics(
                        gcp_points=self.window.gcp_points,
                        veri_points=self.window.verification_points,
                        polygon_coords=self.window.polygon_coords,
                        utm_crs=crs,
                        elevation_data=self.window.elevation_data,
                        detection_score=detection_score
                    )
                    overlay_success = True
                    self.window.logger.info(f"Generated overlay diagnostics at {self.window.diagnostic_report}")
                except Exception as e:
                    self.window.logger.warning(f"Overlay generation failed: {e}")
                    self.status.emit(f"Warning: Overlay generation failed ({e}); continuing without overlay")
                    overlay_path = None
            else:
                self.window.logger.warning("No DSM or DEM available for overlay generation")
                self.status.emit("Warning: No DSM/DEM available; export will proceed without overlay")
                overlay_path = None
            flight_params = self.window.main_logic.flight_params.copy()
            flight_params['prr_khz'] = float(self.window.prr_input.text()) if self.window.prr_input.text() else 640.0
            elevations = self.window.elevation_data.get('profile', [])
            if not elevations or all(e <= 0 for e in elevations):
                self.window.logger.warning("No valid elevations for report; using fallback")
                latlon_points = [(p['lat'], p['lon']) for p in self.window.gcp_points + self.window.verification_points]
                try:
                    elevations, _ = get_srtm_elevation_bulk(latlon_points, retry=3)
                    valid_elevations = [float(e) for e in elevations if e is not None and e > 0]
                    if not valid_elevations or len(valid_elevations) == 0:
                        self.window.logger.warning("OpenTopoData fallback yielded no valid elevations")
                        elevations = [pt.get('elevation', self.window.main_logic.centroid_elevation) for pt in self.window.gcp_points + self.window.verification_points]
                    else:
                        self.window.elevation_data['profile'] = valid_elevations
                        self.window.elevation_data['min'] = float(min(valid_elevations))
                        self.window.elevation_data['max'] = float(max(valid_elevations))
                        self.window.elevation_data['mean'] = float(sum(valid_elevations) / len(valid_elevations))
                except Exception as e:
                    self.window.logger.warning(f"Fallback elevation fetch failed: {e}")
                    elevations = [pt.get('elevation', self.window.main_logic.centroid_elevation) for pt in self.window.gcp_points + self.window.verification_points]
                    self.window.elevation_data['profile'] = elevations
                    self.window.elevation_data['min'] = float(min(elevations)) if elevations and len(elevations) > 0 else self.window.main_logic.centroid_elevation
                    self.window.elevation_data['max'] = float(max(elevations)) if elevations and len(elevations) > 0 else self.window.main_logic.centroid_elevation
                    self.window.elevation_data['mean'] = float(sum(elevations) / len(elevations)) if elevations and len(elevations) > 0 else self.window.main_logic.centroid_elevation
            create_survey_summary_pdf(
                output_path=report_path,
                polygon_coords=self.window.polygon_coords,
                gcp_points=self.window.gcp_points,
                veri_points=self.window.verification_points,
                spacing=float(self.window.elevation_data.get('recommended_spacing_m', 10.0)),
                layout_mode=self.window.main_logic.layout_mode,
                area_acres=self.window.main_logic.polygon_area_acres or 0.0,
                elevation_data={
                    'elevations': elevations,
                    'cycle_score': self.window.elevation_data.get('cycle_score', 0.0),
                    'pattern_score': self.window.elevation_data.get('pattern_score', 0.0),
                    'suggested_layout': self.window.elevation_data.get('suggested_layout', 'grid'),
                    'recommended_spacing_m': self.window.elevation_data.get('recommended_spacing_m', 10.0)
                },
                overlay_image_path=overlay_path if overlay_success else None,
                flight_timings=None,
                pattern_score=self.window.elevation_data.get("pattern_score", 0.0),
                project_name=self.window.area_name or "UAS Survey",
                logo_path=resource_path("app_icon.png") if os.path.exists(resource_path("app_icon.png")) else None,
                operator_name="RPIC",
                survey_location=None,
                notes="Conducted under clear skies with no precipitation.",
                kmz_link=self.window.kmz_input.text(),
                model_3d_path=self.window.main_logic.generate_3d_model(elevations, self.window.polygon_coords) if elevations and len(elevations) > 0 else None,
                flight_params=flight_params,
                rejected_gcp_points=self.window.rejected_gcp_points,
                rejected_veri_points=self.window.rejected_verification_points
            )
            self.window.logger.info(f"Exported report to {report_path}")
            zip_path = os.path.join(self.output_dir, f"{project_name}_{timestamp}_deliverables.zip")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(csv_path, os.path.basename(csv_path))
                zipf.write(kml_path, os.path.basename(kml_path))
                zipf.write(report_path, os.path.basename(report_path))
                if overlay_success and os.path.exists(overlay_path):
                    zipf.write(overlay_path, os.path.basename(overlay_path))
                    self.window.logger.info(f"Included overlay map in ZIP: {overlay_path}")
                else:
                    self.window.logger.warning(f"Overlay map not included in ZIP: {overlay_path}")
            self.finished.emit(zip_path)
        except Exception as e:
            self.error.emit(f"Export failed: {str(e)}")
            self.window.logger.error(f"ExportDeliverablesThread failed: {str(e)}")

class UASWindow(QMainWindow):
    status_update = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAS Survey Tool v2.0")
        icon_path = resource_path("app_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.main_logic = MainLogic(parent=self, cuda_enabled=PYCUDA_AVAILABLE)
        self.gcp_points = []
        self.verification_points = []
        self.rejected_gcp_points = []
        self.rejected_verification_points = []
        self.flight_params = {}
        self.area_name = ""
        self.polygon_coords = []
        self.elevation_data = {}
        self.diagnostic_report = {}
        self.kmz_loaded = False
        self.last_kmz_load_time = 0
        self.debounce_interval = 1.0
        self.setup_logging()
        self.init_ui()
        self.status_update.connect(self.status_bar.showMessage)
        self.logger.debug(f"UASWindow instance created with cuda_enabled={PYCUDA_AVAILABLE}")

    def setup_logging(self):
        logging.getLogger().handlers = []
        logging.getLogger(__name__).handlers = []
        log_dir = os.path.expanduser("~/UAS_Survey_Tool_Logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "uas_survey_tool.log")
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding='utf-8')]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.logger.debug(f"Logger handlers after setup: {len(self.logger.handlers)}")

    def init_ui(self):
        screen = QApplication.primaryScreen()
        screen_size = screen.availableGeometry()
        screen_width = screen_size.width()
        screen_height = screen_size.height()
        min_width = max(800, int(screen_width * 0.7))
        min_height = max(600, int(screen_height * 0.7))
        max_width = min(1400, int(screen_width * 0.9))
        max_height = min(1000, int(screen_height * 0.9))
        self.setMinimumSize(min_width, min_height)
        self.setMaximumSize(max_width, max_height)
        self.resize(min_width, 800)
        self.logger.debug(f"Screen size: {screen_width}x{screen_height}, "
                         f"Min size: {min_width}x{min_height}, "
                         f"Max size: {max_width}x{max_height}")
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea { border: none; background-color: #1e2a44; }
            QScrollBar:vertical { border: none; background: #2a3f73; width: 10px; margin: 0px; }
            QScrollBar::handle:vertical { background: #3f5ca8; border-radius: 5px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { background: none; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        """)
        self.setCentralWidget(scroll_area)
        main_widget = QWidget()
        scroll_area.setWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        font = QFont("Helvetica", 12)
        self.setFont(font)
        try:
            style_path = resource_path("styles.qss")
            if os.path.exists(style_path):
                with open(style_path, "r", encoding='utf-8') as f:
                    self.setStyleSheet(f.read())
                self.logger.debug("Stylesheet loaded successfully.")
            else:
                self.logger.error(f"Stylesheet not found at {style_path}")
                self.status_bar.showMessage(f"Stylesheet not found at {style_path}")
                QMessageBox.warning(self, "Warning", f"Stylesheet not found at {style_path}. Using default styling.")
        except Exception as e:
            self.logger.error(f"Failed to load stylesheet: {e}")
            self.status_bar.showMessage("Failed to load stylesheet")
            QMessageBox.critical(self, "Error", f"Failed to load stylesheet: {e}")
        input_group = QGroupBox("Input Parameters")
        input_layout = QFormLayout()
        input_layout.setSpacing(10)
        input_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        input_layout.setContentsMargins(10, 20, 10, 10)
        self.kmz_label = QLabel("KMZ/KML File:")
        self.kmz_label.setToolTip("Select a KMZ/KML file defining the Area of Interest (AOI).")
        kmz_widget = QHBoxLayout()
        self.kmz_input = QLineEdit()
        self.kmz_input.setReadOnly(True)
        self.kmz_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.kmz_input.setMinimumHeight(30)
        self.kmz_button = QPushButton("Browse")
        self.kmz_button.setToolTip("Browse for a KMZ/KML file.")
        self.kmz_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.kmz_button.setMinimumWidth(80)
        self.kmz_button.setMinimumHeight(30)
        self.kmz_button.clicked.connect(self.browse_kmz)
        kmz_widget.addWidget(self.kmz_input)
        kmz_widget.addWidget(self.kmz_button)
        input_layout.addRow(self.kmz_label, kmz_widget)
        self.centroid_label = QLabel("Centroid (Lon, Lat):")
        self.centroid_display = QLineEdit()
        self.centroid_display.setReadOnly(True)
        self.centroid_display.setToolTip("Centroid coordinates of the loaded AOI.")
        input_layout.addRow(self.centroid_label, self.centroid_display)
        self.dem_check = QCheckBox("Auto-Download DEM")
        self.dem_check.setChecked(True)
        self.dem_check.setEnabled(True)
        self.dem_check.setToolTip("Toggle DEM auto-download.")
        input_layout.addRow(self.dem_check)
        self.dem_label = QLabel("DEM File:")
        self.dem_label.setToolTip("DEM file path (auto-downloaded or manual).")
        dem_widget = QHBoxLayout()
        self.dem_input = QLineEdit()
        self.dem_input.setReadOnly(True)
        self.dem_input.setEnabled(True)
        self.dem_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.dem_input.setMinimumHeight(30)
        self.dem_button = QPushButton("Browse")
        self.dem_button.setToolTip("Browse for a DEM file.")
        self.dem_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.dem_button.setMinimumWidth(80)
        self.dem_button.setMinimumHeight(30)
        self.dem_button.clicked.connect(self.browse_dem)
        dem_widget.addWidget(self.dem_input)
        dem_widget.addWidget(self.dem_button)
        input_layout.addRow(self.dem_label, dem_widget)
        self.dsm_check = QCheckBox("Auto-Download DSM")
        self.dsm_check.setChecked(True)
        self.dsm_check.setEnabled(True)
        self.dsm_check.setToolTip("Toggle DSM auto-download.")
        input_layout.addRow(self.dsm_check)
        self.dsm_label = QLabel("DSM File:")
        self.dsm_label.setToolTip("DSM file path (auto-downloaded or manual).")
        dsm_widget = QHBoxLayout()
        self.dsm_input = QLineEdit()
        self.dsm_input.setReadOnly(True)
        self.dsm_input.setEnabled(True)
        self.dsm_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.dsm_input.setMinimumHeight(30)
        self.dsm_button = QPushButton("Browse")
        self.dsm_button.setToolTip("Browse for a DSM file.")
        self.dsm_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.dsm_button.setMinimumWidth(80)
        self.dsm_button.setMinimumHeight(30)
        self.dsm_button.clicked.connect(self.browse_dsm)
        dsm_widget.addWidget(self.dsm_input)
        dsm_widget.addWidget(self.dsm_button)
        input_layout.addRow(self.dsm_label, dsm_widget)
        self.spacing_label = QLabel("GCP Spacing (m):")
        self.spacing_label.setToolTip("Spacing is auto-calculated based on accuracy.")
        self.spacing_input = QLineEdit()
        self.spacing_input.setValidator(QDoubleValidator(1.0, 1000.0, 2))
        self.spacing_input.setPlaceholderText("Auto-calculated")
        self.spacing_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.spacing_input.setMinimumHeight(30)
        input_layout.addRow(self.spacing_label, self.spacing_input)
        self.weights_label = QLabel("Weights (recursive,cyclic,fractal,chaotic):")
        self.weights_label.setToolTip("Enter 4 comma-separated weights for point scoring (default: 0.25,0.25,0.25,0.25).")
        self.weights_input = QLineEdit("0.25,0.25,0.25,0.25")
        self.weights_input.setPlaceholderText("0.25,0.25,0.25,0.25")
        self.weights_input.setToolTip("Enter 4 comma-separated weights for point scoring.")
        self.weights_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.weights_input.setMinimumHeight(30)
        input_layout.addRow(self.weights_label, self.weights_input)
        self.layout_label = QLabel("Layout Mode:")
        self.layout_label.setToolTip("Layout is auto-selected based on terrain analysis.")
        self.layout_combo = QComboBox()
        layout_patterns = [
            ("Grid", "Places GCPs in a uniform grid."),
            ("Triangular", "Arranges GCPs in a triangular lattice."),
            ("Diamond", "Uses a rotated grid for balanced distribution."),
            ("Spiral", "Distributes GCPs in a spiral pattern."),
            ("Fractal", "Applies recursive Fibonacci placement."),
            ("Chaotic", "Places GCPs in a pseudo-random chaotic pattern.")
        ]
        for pattern, tooltip in layout_patterns:
            self.layout_combo.addItem(pattern)
            self.layout_combo.setItemData(self.layout_combo.count()-1, tooltip, Qt.ItemDataRole.ToolTipRole)
        self.layout_combo.setCurrentText("Grid")
        self.layout_combo.setEnabled(True)
        self.layout_combo.setToolTip("Select layout mode for GCP placement.")
        self.layout_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.layout_combo.setMinimumHeight(30)
        input_layout.addRow(self.layout_label, self.layout_combo)
        self.sequence_label = QLabel("Sequence Type:")
        self.sequence_label.setToolTip("Select sequence type for GCP placement.")
        self.sequence_combo = QComboBox()
        sequence_types = [
            ("fibonacci", "Uses Fibonacci sequence for balanced distribution."),
            ("lucas", "Applies Lucas sequence for varied spacing."),
            ("tribonacci", "Employs Tribonacci sequence for denser patterns."),
            ("pell", "Uses Pell sequence for robust distribution.")
        ]
        for seq_type, tooltip in sequence_types:
            self.sequence_combo.addItem(seq_type)
            self.sequence_combo.setItemData(self.sequence_combo.count()-1, tooltip, Qt.ItemDataRole.ToolTipRole)
        self.sequence_combo.setCurrentText("fibonacci")
        self.sequence_combo.setEnabled(True)
        self.sequence_combo.setToolTip("Select sequence type for GCP placement.")
        self.sequence_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.sequence_combo.setMinimumHeight(30)
        input_layout.addRow(self.sequence_label, self.sequence_combo)
        self.modulus_label = QLabel("Modulus:")
        self.modulus_label.setToolTip("Modulus is auto-calculated.")
        self.modulus_input = QLineEdit()
        self.modulus_input.setReadOnly(True)
        self.modulus_input.setPlaceholderText("Auto-calculated")
        self.modulus_input.setToolTip("Modulus is auto-calculated.")
        self.modulus_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.modulus_input.setMinimumHeight(30)
        input_layout.addRow(self.modulus_label, self.modulus_input)
        self.survey_method_label = QLabel("Survey Method:")
        self.survey_method_label.setToolTip("Toggle between LiDAR and Photogrammetry density logic.")
        self.survey_method_combo = QComboBox()
        survey_methods = [
            ("LiDAR", "Optimized for LiDAR surveys with 15-20 GCPs and 25-30 VCPs."),
            ("Photogrammetry", "Optimized for photogrammetry with 50-100 GCPs and 75-150 VCPs.")
        ]
        for method, tooltip in survey_methods:
            self.survey_method_combo.addItem(method)
            self.survey_method_combo.setItemData(self.survey_method_combo.count()-1, tooltip, Qt.ItemDataRole.ToolTipRole)
        self.survey_method_combo.setCurrentText("LiDAR")
        self.survey_method_combo.setEnabled(True)
        self.survey_method_combo.setToolTip("Select survey method.")
        self.survey_method_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.survey_method_combo.setMinimumHeight(30)
        input_layout.addRow(self.survey_method_label, self.survey_method_combo)
        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)
        flight_group = QGroupBox("Flight Parameters")
        flight_layout = QFormLayout()
        flight_layout.setSpacing(10)
        flight_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        flight_layout.setContentsMargins(10, 20, 10, 10)
        self.accuracy_label = QLabel("Desired Accuracy (mm):")
        self.accuracy_label.setToolTip("Target accuracy for the survey (default 60.96mm).")
        self.accuracy_input = QLineEdit("60.96")
        self.accuracy_input.setValidator(QDoubleValidator(0.01, 999.99, 2))
        self.accuracy_input.setToolTip("Enter accuracy in mm (0.01-999.99).")
        self.accuracy_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.accuracy_input.setMinimumHeight(30)
        flight_layout.addRow(self.accuracy_label, self.accuracy_input)
        self.altitude_label = QLabel("Flight Altitude (m):")
        self.altitude_label.setToolTip("Altitude Above Ground Level (default 80m).")
        self.altitude_input = QLineEdit("80")
        self.altitude_input.setValidator(QDoubleValidator(1.0, 1000.0, 1))
        self.altitude_input.setToolTip("Enter altitude in meters (1-1000).")
        self.altitude_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.altitude_input.setMinimumHeight(30)
        flight_layout.addRow(self.altitude_label, self.altitude_input)
        self.fov_label = QLabel("LiDAR FOV (deg):")
        self.fov_label.setToolTip("Field of View for the LiDAR sensor (default 45°).")
        self.fov_input = QLineEdit("45")
        self.fov_input.setValidator(QDoubleValidator(1.0, 180.0, 1))
        self.fov_input.setToolTip("Enter FOV in degrees (1-180).")
        self.fov_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.fov_input.setMinimumHeight(30)
        flight_layout.addRow(self.fov_label, self.fov_input)
        self.prr_label = QLabel("Pulse Repetition Rate (kHz):")
        self.prr_label.setToolTip("Pulse rate for RECON-XT LiDAR (default 640kHz).")
        self.prr_input = QLineEdit("640")
        self.prr_input.setValidator(QDoubleValidator(1.0, 1000.0, 1))
        self.prr_input.setToolTip("Enter PRR in kHz (1-1000).")
        self.prr_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.prr_input.setMinimumHeight(30)
        flight_layout.addRow(self.prr_label, self.prr_input)
        self.flight_params_display = QTextEdit()
        self.flight_params_display.setReadOnly(True)
        self.flight_params_display.setMinimumHeight(100)
        flight_layout.addRow(QLabel("Flight Parameters:"), self.flight_params_display)
        flight_group.setLayout(flight_layout)
        main_layout.addWidget(flight_group)
        button_row = QHBoxLayout()
        self.review_button = QPushButton("Review Points")
        self.review_button.setObjectName("actionButton")
        self.review_button.setToolTip("Review and edit GCPs and verification points.")
        self.review_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.review_button.setMinimumWidth(120)
        self.review_button.setMinimumHeight(35)
        self.review_button.clicked.connect(self.review_points)
        self.review_button.setEnabled(False)
        self.export_button = QPushButton("Export Deliverables")
        self.export_button.setObjectName("actionButton")
        self.export_button.setToolTip("Export GCPs, verification points, survey plan, report, and overlay map.")
        self.export_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.export_button.setMinimumWidth(120)
        self.export_button.setMinimumHeight(35)
        self.export_button.clicked.connect(self.export_deliverables)
        self.export_button.setEnabled(False)
        button_row.addWidget(self.review_button)
        button_row.addWidget(self.export_button)
        button_row.addStretch()
        main_layout.addLayout(button_row)
        log_label = QLabel("Status Log:")
        log_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        main_layout.addWidget(log_label)
        self.status_log = QTextEdit()
        self.status_log.setReadOnly(True)
        self.status_log.setMinimumHeight(200)
        self.status_log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        main_layout.addWidget(self.status_log)
        handler = TextEditHandler(self.status_log)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        self.logger.addHandler(handler)
        self.logger.debug(f"Logger handlers after adding TextEditHandler: {len(self.logger.handlers)}")
        main_layout.setStretch(0, 5)
        main_layout.setStretch(1, 2)
        main_layout.setStretch(2, 1)
        main_layout.setStretch(4, 2)

    def browse_kmz(self):
        current_time = time.time()
        self.logger.debug(f"browse_kmz called at {current_time}")
        if current_time - self.last_kmz_load_time < self.debounce_interval:
            self.logger.debug("KMZ load debounced")
            return
        file_name, _ = QFileDialog.getOpenFileName(self, "Select KMZ/KML File", "", "KMZ/KML Files (*.kmz *.kml)")
        if file_name:
            self.kmz_input.setText(file_name)
            self.area_name = os.path.splitext(os.path.basename(file_name))[0]
            try:
                self.kmz_loaded = self.main_logic.load_kmz(file_name)
            except Exception as e:
                def show_error():
                    self.status_update.emit(f"Failed to load KMZ: {str(e)}")
                    QMessageBox.critical(self, "Error", f"Failed to load KMZ: {str(e)}")
                if QThread.currentThread() is QApplication.instance().thread():
                    show_error()
                else:
                    QTimer.singleShot(0, show_error)
                self.logger.error(f"Failed to load KMZ: {file_name} with error: {str(e)}")
                return
            if self.kmz_loaded:
                self.polygon_coords = self.main_logic.polygon_coords
                centroid = self.main_logic.centroid
                def update_ui():
                    if centroid:
                        lon, lat = centroid
                        self.centroid_display.setText(f"{lon:.6f}, {lat:.6f}")
                        self.status_update.emit(f"Loaded KMZ: {file_name}, Centroid: ({lon:.6f}, {lat:.6f})")
                    else:
                        self.status_update.emit(f"Loaded KMZ: {file_name}, Centroid not available")
                    self.generate_plan()
                if QThread.currentThread() is QApplication.instance().thread():
                    update_ui()
                else:
                    QTimer.singleShot(0, update_ui)
                self.last_kmz_load_time = current_time
                self.logger.info(f"Successfully loaded KMZ: {file_name}")
            else:
                def show_error():
                    self.status_update.emit("Failed to load KMZ file")
                    QMessageBox.critical(self, "Error", "Failed to load KMZ file")
                if QThread.currentThread() is QApplication.instance().thread():
                    show_error()
                else:
                    QTimer.singleShot(0, show_error)
                self.logger.error(f"Failed to load KMZ: {file_name}")

    def browse_dem(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select DEM File", "", "GeoTIFF Files (*.tif *.tiff)")
        if file_name:
            if os.path.exists(file_name):
                def update_ui():
                    self.dem_input.setText(file_name)
                    self.main_logic.dem_path = file_name
                    self.main_logic.centroid_elevation = 291.0
                    self.status_update.emit(f"Selected DEM file: {file_name}")
                if QThread.currentThread() is QApplication.instance().thread():
                    update_ui()
                else:
                    QTimer.singleShot(0, update_ui)
                self.logger.info(f"Selected DEM file: {file_name}")
            else:
                def show_error():
                    self.status_update.emit(f"Selected DEM file does not exist: {file_name}")
                    QMessageBox.critical(self, "Error", f"Selected DEM file does not exist: {file_name}")
                if QThread.currentThread() is QApplication.instance().thread():
                    show_error()
                else:
                    QTimer.singleShot(0, show_error)
                self.logger.error(f"Selected DEM file does not exist: {file_name}")

    def browse_dsm(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select DSM File", "", "GeoTIFF Files (*.tif *.tiff)")
        if file_name:
            if os.path.exists(file_name):
                def update_ui():
                    self.dsm_input.setText(file_name)
                    self.main_logic.dsm_path = file_name
                    self.status_update.emit(f"Selected DSM file: {file_name}")
                if QThread.currentThread() is QApplication.instance().thread():
                    update_ui()
                else:
                    QTimer.singleShot(0, update_ui)
                self.logger.info(f"Selected DSM file: {file_name}")
            else:
                def show_error():
                    self.status_update.emit(f"Selected DSM file does not exist: {file_name}")
                    QMessageBox.critical(self, "Error", f"Selected DSM file does not exist: {file_name}")
                if QThread.currentThread() is QApplication.instance().thread():
                    show_error()
                else:
                    QTimer.singleShot(0, show_error)
                self.logger.error(f"Selected DSM file does not exist: {file_name}")

    def generate_plan(self):
        if not self.kmz_loaded:
            def show_error():
                self.logger.error("No KMZ file loaded; cannot generate plan")
                self.status_update.emit("No KMZ file loaded")
                QMessageBox.critical(self, "Error", "Please load a KMZ/KML file first.")
            if QThread.currentThread() is QApplication.instance().thread():
                show_error()
            else:
                QTimer.singleShot(0, show_error)
            return
        dsm_or_dem_path = getattr(self.main_logic, "dsm_path", None) or getattr(self.main_logic, "dem_path", None)
        if not dsm_or_dem_path and not (self.dem_check.isChecked() or self.dsm_check.isChecked()):
            def show_error():
                self.logger.error("No valid DSM/DEM path and auto-download disabled")
                self.status_update.emit("No valid DSM/DEM; enable auto-download or select a file")
                QMessageBox.critical(self, "Error", "No valid DSM/DEM file. Enable auto-download or select a file.")
            if QThread.currentThread() is QApplication.instance().thread():
                show_error()
            else:
                QTimer.singleShot(0, show_error)
            return
        def start_plan():
            self.status_update.emit("Generating plan, please wait...")
            self.review_button.setEnabled(False)
            self.export_button.setEnabled(False)
            self.plan_thread = PlanGenerationThread(self)
            self.plan_thread.finished.connect(self.on_plan_generated)
            self.plan_thread.error.connect(self.on_plan_error)
            self.plan_thread.status.connect(self.status_update)
            self.plan_thread.start()
        if QThread.currentThread() is QApplication.instance().thread():
            start_plan()
        else:
            QTimer.singleShot(0, start_plan)

    def on_plan_generated(self, result):
        gcp_points = result.get('gcp_points', [])
        verification_points = result.get('verification_points', [])
        self.rejected_gcp_points = result.get('rejected_gcp_points', [])
        self.rejected_verification_points = result.get('rejected_verification_points', [])
        self.elevation_data = result.get('elevation_data', {})
        if not gcp_points or len(gcp_points) == 0:
            def show_error():
                self.logger.error("No GCPs generated")
                self.status_update.emit("Failed to generate GCP plan")
                QMessageBox.critical(self, "Error", "Failed to generate GCP plan.")
            if QThread.currentThread() is QApplication.instance().thread():
                show_error()
            else:
                QTimer.singleShot(0, show_error)
            return
        self.gcp_points = gcp_points
        self.verification_points = verification_points
        elevations = self.elevation_data.get('profile', [])
        dsm_or_dem_path = getattr(self.main_logic, "dsm_path", None) or getattr(self.main_logic, "dem_path", None)
        if not elevations or all(e <= 0 for e in elevations):
            self.logger.warning("No valid elevations; attempting fallback")
            latlon_points = [(p['lat'], p['lon']) for p in self.gcp_points + self.verification_points]
            if dsm_or_dem_path and os.path.exists(dsm_or_dem_path):
                try:
                    elevations, cycle_score, pattern_score = get_dsm_elevation_profile(
                        latlon_points, raster_path=dsm_or_dem_path, use_dsm=True
                    )
                    valid_elevations = [float(e) for e in elevations if e is not None and e > 0]
                    if valid_elevations and len(valid_elevations) > 0:
                        self.elevation_data['profile'] = valid_elevations
                        self.elevation_data['cycle_score'] = float(cycle_score)
                        self.elevation_data['pattern_score'] = float(pattern_score)
                        self.elevation_data['suggested_layout'] = suggest_layout_based_on_dsm(
                            valid_elevations,
                            self.main_logic.polygon_area_acres * 4046.86 if self.main_logic.polygon_area_acres else 0.0,
                            cycle_score,
                            pattern_score
                        )
                        self.elevation_data['min'] = float(min(valid_elevations))
                        self.elevation_data['max'] = float(max(valid_elevations))
                        self.elevation_data['mean'] = float(sum(valid_elevations) / len(valid_elevations))
                    else:
                        self.logger.warning("DSM/DEM elevation refresh yielded no valid elevations")
                        try:
                            elevations, _ = get_srtm_elevation_bulk(latlon_points, retry=3)
                            valid_elevations = [float(e) for e in elevations if e is not None and e > 0]
                            if valid_elevations and len(valid_elevations) > 0:
                                self.elevation_data['profile'] = valid_elevations
                                self.elevation_data['min'] = float(min(valid_elevations))
                                self.elevation_data['max'] = float(max(valid_elevations))
                                self.elevation_data['mean'] = float(sum(valid_elevations) / len(valid_elevations))
                            else:
                                self.logger.warning("OpenTopoData fallback yielded no valid elevations")
                                elevations = [pt.get('elevation', self.main_logic.centroid_elevation) for pt in self.gcp_points + self.verification_points]
                                self.elevation_data['profile'] = elevations
                                self.elevation_data['min'] = float(min(elevations)) if elevations and len(elevations) > 0 else self.main_logic.centroid_elevation
                                self.elevation_data['max'] = float(max(elevations)) if elevations and len(elevations) > 0 else self.main_logic.centroid_elevation
                                self.elevation_data['mean'] = float(sum(elevations) / len(elevations)) if elevations and len(elevations) > 0 else self.main_logic.centroid_elevation
                        except Exception as e:
                            self.logger.warning(f"OpenTopoData fallback failed: {e}")
                            elevations = [pt.get('elevation', self.main_logic.centroid_elevation) for pt in self.gcp_points + self.verification_points]
                            self.elevation_data['profile'] = elevations
                            self.elevation_data['min'] = float(min(elevations)) if elevations and len(elevations) > 0 else self.main_logic.centroid_elevation
                            self.elevation_data['max'] = float(max(elevations)) if elevations and len(elevations) > 0 else self.main_logic.centroid_elevation
                            self.elevation_data['mean'] = float(sum(elevations) / len(elevations)) if elevations and len(elevations) > 0 else self.main_logic.centroid_elevation
                except Exception as e:
                    self.logger.warning(f"Failed to refresh elevation data from DSM/DEM: {e}")
                    self.status_update.emit(f"Failed to refresh elevation data: {e}")
                    try:
                        elevations, _ = get_srtm_elevation_bulk(latlon_points, retry=3)
                        valid_elevations = [float(e) for e in elevations if e is not None and e > 0]
                        if valid_elevations and len(valid_elevations) > 0:
                            self.elevation_data['profile'] = valid_elevations
                            self.elevation_data['min'] = float(min(valid_elevations))
                            self.elevation_data['max'] = float(max(valid_elevations))
                            self.elevation_data['mean'] = float(sum(valid_elevations) / len(valid_elevations))
                        else:
                            self.logger.warning("OpenTopoData fallback yielded no valid elevations")
                            elevations = [pt.get('elevation', self.main_logic.centroid_elevation) for pt in self.gcp_points + self.verification_points]
                            self.elevation_data['profile'] = elevations
                            self.elevation_data['min'] = float(min(elevations)) if elevations and len(elevations) > 0 else self.main_logic.centroid_elevation
                            self.elevation_data['max'] = float(max(elevations)) if elevations and len(elevations) > 0 else self.main_logic.centroid_elevation
                            self.elevation_data['mean'] = float(sum(elevations) / len(elevations)) if elevations and len(elevations) > 0 else self.main_logic.centroid_elevation
                    except Exception as e:
                        self.logger.warning(f"OpenTopoData fallback failed: {e}")
                        elevations = [pt.get('elevation', self.main_logic.centroid_elevation) for pt in self.gcp_points + self.verification_points]
                        self.elevation_data['profile'] = elevations
                        self.elevation_data['min'] = float(min(elevations)) if elevations and len(elevations) > 0 else self.main_logic.centroid_elevation
                        self.elevation_data['max'] = float(max(elevations)) if elevations and len(elevations) > 0 else self.main_logic.centroid_elevation
                        self.elevation_data['mean'] = float(sum(elevations) / len(elevations)) if elevations and len(elevations) > 0 else self.main_logic.centroid_elevation
            else:
                self.logger.warning("No DSM/DEM path; using fallback elevations")
                try:
                    elevations, _ = get_srtm_elevation_bulk(latlon_points, retry=3)
                    valid_elevations = [float(e) for e in elevations if e is not None and e > 0]
                    if valid_elevations and len(valid_elevations) > 0:
                        self.elevation_data['profile'] = valid_elevations
                        self.elevation_data['min'] = float(min(valid_elevations))
                        self.elevation_data['max'] = float(max(valid_elevations))
                        self.elevation_data['mean'] = float(sum(valid_elevations) / len(valid_elevations))
                    else:
                        self.logger.warning("OpenTopoData fallback yielded no valid elevations")
                        elevations = [pt.get('elevation', self.main_logic.centroid_elevation) for pt in self.gcp_points + self.verification_points]
                        self.elevation_data['profile'] = elevations
                        self.elevation_data['min'] = float(min(elevations)) if elevations and len(elevations) > 0 else self.main_logic.centroid_elevation
                        self.elevation_data['max'] = float(max(elevations)) if elevations and len(elevations) > 0 else self.main_logic.centroid_elevation
                        self.elevation_data['mean'] = float(sum(elevations) / len(elevations)) if elevations and len(elevations) > 0 else self.main_logic.centroid_elevation
                except Exception as e:
                    self.logger.warning(f"OpenTopoData fallback failed: {e}")
                    elevations = [pt.get('elevation', self.main_logic.centroid_elevation) for pt in self.gcp_points + self.verification_points]
                    self.elevation_data['profile'] = elevations
                    self.elevation_data['min'] = float(min(elevations)) if elevations and len(elevations) > 0 else self.main_logic.centroid_elevation
                    self.elevation_data['max'] = float(max(elevations)) if elevations and len(elevations) > 0 else self.main_logic.centroid_elevation
                    self.elevation_data['mean'] = float(sum(elevations) / len(elevations)) if elevations and len(elevations) > 0 else self.main_logic.centroid_elevation
        self.logger.info(f"Elevation data: min={self.elevation_data.get('min', 0.0):.2f}, "
                        f"max={self.elevation_data.get('max', 0.0):.2f}, "
                        f"mean={self.elevation_data.get('mean', 0.0):.2f}, "
                        f"count={len(self.elevation_data.get('profile', []))}")
        if len(self.elevation_data.get('profile', [])) > 0 and max(self.elevation_data.get('profile', [0])) <= 0:
            self.logger.warning("All elevations are zero or invalid; check DSM/DEM data or API connectivity")
        overlay_success = False
        crs = getattr(self.main_logic, 'crs', None)
        if not crs and self.polygon_coords and len(self.polygon_coords) > 0:
            centroid_lon = sum(lon for lon, _ in self.polygon_coords) / len(self.polygon_coords)
            centroid_lat = sum(lat for _, lat in self.polygon_coords) / len(self.polygon_coords)
            crs = auto_detect_utm_crs(centroid_lon, centroid_lat)
            self.main_logic.crs = crs
            self.logger.debug(f"Set CRS from centroid: {crs}")
        elif not crs:
            crs = 'EPSG:32614'
            self.main_logic.crs = crs
            self.logger.warning("No polygon coordinates available; using default CRS: EPSG:32614")
        if dsm_or_dem_path and os.path.exists(dsm_or_dem_path):
            try:
                detection_score = self.elevation_data.get('pattern_score', 0.0)
                self.diagnostic_report = generate_overlay_diagnostics(
                    gcp_points=self.gcp_points,
                    veri_points=self.verification_points,
                    polygon_coords=self.polygon_coords,
                    utm_crs=crs,
                    elevation_data=self.elevation_data,
                    detection_score=detection_score
                )
                overlay_success = True
                self.logger.info(f"Generated overlay diagnostics at {self.diagnostic_report}")
            except Exception as e:
                self.logger.warning(f"Overlay generation failed: {e}")
                def show_warning():
                    self.status_update.emit(f"Warning: Overlay generation failed ({e}); export can proceed without overlay")
                if QThread.currentThread() is QApplication.instance().thread():
                    show_warning()
                else:
                    QTimer.singleShot(0, show_warning)
                self.diagnostic_report = {"output": None, "elevation": {}}
        else:
            self.logger.warning("No DSM or DEM available for overlay generation")
            def show_warning():
                self.status_update.emit("Warning: No DSM/DEM available; export will proceed without overlay")
            if QThread.currentThread() is QApplication.instance().thread():
                show_warning()
            else:
                QTimer.singleShot(0, show_warning)
            self.diagnostic_report = {"output": None, "elevation": {}}
        def update_ui():
            if not self.spacing_input.text():
                self.spacing_input.setText(f"{self.elevation_data.get('recommended_spacing_m', 10.0):.2f}")
            self.flight_params_display.setText(
                f"Accuracy: {self.main_logic.flight_params.get('accuracy', 60.96):.2f} mm\n"
                f"Altitude: {self.main_logic.flight_params.get('altitude_m', 80.0):.1f} m\n"
                f"FOV: {self.main_logic.flight_params.get('fov', 45.0):.1f} deg\n"
                f"PRR: {self.main_logic.flight_params.get('prr_khz', 640.0):.1f} kHz\n"
                f"Speed: {self.main_logic.flight_params.get('speed_mps', 10.0):.1f} m/s\n"
                f"GSD: {self.main_logic.flight_params.get('gsd_mm', 60.96):.2f} mm"
            )
            self.review_button.setEnabled(True)
            self.export_button.setEnabled(True)
            self.status_update.emit(f"GCP plan generated: {len(self.gcp_points)} GCPs, {len(self.verification_points)} VCPs")
        if QThread.currentThread() is QApplication.instance().thread():
            update_ui()
        else:
            QTimer.singleShot(0, update_ui)
        self.logger.info(f"Generated plan with {len(self.gcp_points)} GCPs, {len(self.verification_points)} VCPs")

    def on_plan_error(self, error):
        def show_error():
            self.logger.error(f"Plan generation failed: {error}")
            self.status_update.emit("Plan generation failed")
            QMessageBox.critical(self, "Error", f"Plan generation failed: {error}")
            self.review_button.setEnabled(False)
            self.export_button.setEnabled(False)
        if QThread.currentThread() is QApplication.instance().thread():
            show_error()
        else:
            QTimer.singleShot(0, show_error)

    def review_points(self):
        try:
            area_m2 = (self.main_logic.polygon_area_acres or 0.0) * 4046.86
            dialog = PointEditorDialog(
                self.main_logic,
                self.gcp_points,
                self.verification_points,
                area_m2,
                parent=self
            )
            def update_points():
                if dialog.exec():
                    self.gcp_points = dialog.gcp_points
                    self.verification_points = dialog.verification_points
                    self.rejected_gcp_points = dialog.rejected_gcp_points
                    self.rejected_verification_points = dialog.rejected_verification_points
                    self.logger.info("Points updated via PointEditorDialog")
                    self.status_update.emit("Points updated successfully")
                else:
                    self.logger.info("Point review cancelled")
                    self.status_update.emit("Point review cancelled")
            if QThread.currentThread() is QApplication.instance().thread():
                update_points()
            else:
                QTimer.singleShot(0, update_points)
        except Exception as e:
            def show_error():
                self.logger.error(f"Failed to open PointEditorDialog: {e}")
                self.status_update.emit("Failed to open point editor")
                QMessageBox.critical(self, "Error", f"Failed to open point editor: {e}")
            if QThread.currentThread() is QApplication.instance().thread():
                show_error()
            else:
                QTimer.singleShot(0, show_error)

    def export_deliverables(self):
        def select_directory():
            output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
            if output_dir:
                self.export_button.setEnabled(False)
                self.export_thread = ExportDeliverablesThread(self, output_dir)
                self.export_thread.finished.connect(self.on_export_finished)
                self.export_thread.error.connect(self.on_export_error)
                self.export_thread.status.connect(self.status_update)
                self.export_thread.overwrite_check.connect(self.handle_overwrite_check)
                self.export_thread.start()
        if QThread.currentThread() is QApplication.instance().thread():
            select_directory()
        else:
            QTimer.singleShot(0, select_directory)

    def handle_overwrite_check(self, csv_path, kml_path, report_path, overlay_path):
        def check_overwrite():
            self.export_button.setEnabled(True)
            existing_files = [f for f in [csv_path, kml_path, report_path, overlay_path] if os.path.exists(f)]
            if len(existing_files) > 0:
                msg = f"The following files already exist:\n{', '.join(existing_files)}\nOverwrite?"
                reply = QMessageBox.question(self, "Overwrite Files?", msg, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.Yes:
                    self.export_button.setEnabled(False)
                    self.export_thread = ExportDeliverablesThread(self, self.export_thread.output_dir)
                    self.export_thread.finished.connect(self.on_export_finished)
                    self.export_thread.error.connect(self.on_export_error)
                    self.export_thread.status.connect(self.status_update)
                    self.export_thread.overwrite_check.connect(self.handle_overwrite_check)
                    self.export_thread.start()
                else:
                    self.logger.info("Export cancelled due to existing files")
                    self.status_update.emit("Export cancelled due to existing files")
            else:
                self.logger.warning("Overwrite check triggered without existing files")
                self.status_update.emit("Export cancelled")
        if QThread.currentThread() is QApplication.instance().thread():
            check_overwrite()
        else:
            QTimer.singleShot(0, check_overwrite)

    def on_export_finished(self, zip_path):
        def update_ui():
            self.export_button.setEnabled(True)
            if zip_path:
                self.logger.info(f"Exported deliverables to {zip_path}")
                self.status_update.emit(f"Deliverables exported to {zip_path}")
                QMessageBox.information(self, "Success", f"Deliverables exported to {zip_path}")
            else:
                self.logger.warning("Export cancelled or failed")
                self.status_update.emit("Export cancelled")
        if QThread.currentThread() is QApplication.instance().thread():
            update_ui()
        else:
            QTimer.singleShot(0, update_ui)

    def on_export_error(self, error):
        def show_error():
            self.export_button.setEnabled(True)
            self.logger.error(f"Export failed: {error}")
            self.status_update.emit("Export failed")
            QMessageBox.critical(self, "Error", f"Export failed: {error}")
        if QThread.currentThread() is QApplication.instance().thread():
            show_error()
        else:
            QTimer.singleShot(0, show_error)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = UASWindow()
    window.show()
    sys.exit(app.exec())