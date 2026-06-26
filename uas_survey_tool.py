import os
import sys
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QGroupBox, QPushButton, QFileDialog,
    QLineEdit, QComboBox, QLabel, QDoubleSpinBox, QCheckBox, QMessageBox,
    QFormLayout, QSpinBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from main_logic import run_export_pipeline
from kmz_parser import extract_geometries_from_kmz
from icon_handler import get_app_icon

class UASSurveyTool(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UAS Survey Tool v2.0")
        self.geometries = []
        self.selected_geometry = None
        self.pattern_score = 0.0
        self.init_ui()
        self.load_stylesheet()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Survey Area Group
        main_layout.addWidget(self.create_kmz_group())

        # Flight Parameters Group
        main_layout.addWidget(self.create_flight_parameters_group())

        # Export Configuration Group
        main_layout.addWidget(self.create_export_options_group())

        # Pattern Score Display
        self.pattern_score_label = QLabel("Pattern Score: N/A")
        main_layout.addWidget(self.pattern_score_label)

        # Export Button
        self.export_btn = QPushButton("Export Deliverables")
        self.export_btn.clicked.connect(self.export_deliverables)
        main_layout.addWidget(self.export_btn)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def create_kmz_group(self):
        group = QGroupBox("Survey Area")
        layout = QFormLayout()

        # KMZ/KML File
        self.kmz_path = QLineEdit()
        self.kmz_path.setPlaceholderText("Select KMZ/KML file")
        self.kmz_path.setReadOnly(True)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_kmz)
        layout.addRow(QLabel("KMZ/KML File:"), self.kmz_path)
        layout.addRow("", browse_btn)

        # Geometry Selection
        self.geometry_combo = QComboBox()
        self.geometry_combo.currentIndexChanged.connect(self.update_selected_geometry)
        layout.addRow(QLabel("Select Geometry:"), self.geometry_combo)

        # Area Display
        self.area_label = QLabel("Survey Area: 0.00 acres")
        layout.addRow(QLabel("Survey Area:"), self.area_label)

        group.setLayout(layout)
        return group

    def create_flight_parameters_group(self):
        group = QGroupBox("Flight Parameters")
        layout = QFormLayout()

        # Altitude
        self.altitude_spin = QDoubleSpinBox()
        self.altitude_spin.setRange(10.0, 400.0)
        self.altitude_spin.setValue(100.0)
        self.altitude_spin.setSuffix(" ft")
        layout.addRow(QLabel("Altitude:"), self.altitude_spin)

        # Speed
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(1.0, 50.0)
        self.speed_spin.setValue(10.0)
        self.speed_spin.setSuffix(" m/s")
        layout.addRow(QLabel("Speed:"), self.speed_spin)

        # Camera FOV
        self.fov_spin = QDoubleSpinBox()
        self.fov_spin.setRange(10.0, 120.0)
        self.fov_spin.setValue(60.0)
        self.fov_spin.setSuffix(" degrees")
        layout.addRow(QLabel("Camera FOV:"), self.fov_spin)

        # Pulse Rate
        self.pulse_spin = QDoubleSpinBox()
        self.pulse_spin.setRange(1.0, 1000000.0)
        self.pulse_spin.setValue(160000.0)
        self.pulse_spin.setSuffix(" Hz")
        layout.addRow(QLabel("Pulse Rate:"), self.pulse_spin)

        # Accuracy
        self.accuracy_spin = QDoubleSpinBox()
        self.accuracy_spin.setRange(0.01, 1000.0)
        self.accuracy_spin.setValue(50.0)
        self.accuracy_spin.setSuffix(" mm")
        layout.addRow(QLabel("Accuracy:"), self.accuracy_spin)

        # Spacing
        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(100.0, 1000.0)
        self.spacing_spin.setValue(400.0)
        self.spacing_spin.setSuffix(" m")
        layout.addRow(QLabel("GCP Spacing:"), self.spacing_spin)

        group.setLayout(layout)
        return group

    def create_export_options_group(self):
        group = QGroupBox("Export Configuration")
        layout = QFormLayout()

        # Export Folder
        self.export_path = QLineEdit()
        self.export_path.setPlaceholderText("Select export folder")
        self.export_path.setReadOnly(True)
        browse_folder_btn = QPushButton("Browse")
        browse_folder_btn.clicked.connect(self.browse_folder)
        layout.addRow(QLabel("Export Folder:"), self.export_path)
        layout.addRow("", browse_folder_btn)

        # GCP Layout
        self.gcp_layout = QComboBox()
        self.gcp_layout.addItems([
            "Grid", "Triangular", "Diamond", "Spiral", "Fibonacci", "Chaotic", "Fractal"
        ])
        layout.addRow(QLabel("GCP Layout:"), self.gcp_layout)
        self.gcp_layout.currentTextChanged.connect(self.update_icon)

        # Pass Count
        self.pass_count_spin = QSpinBox()
        self.pass_count_spin.setRange(1, 10)
        self.pass_count_spin.setValue(1)
        layout.addRow(QLabel("Pass Count:"), self.pass_count_spin)

        # DSM/DEM Selection
        self.dsm_check = QCheckBox("Use Custom DEM")
        self.dsm_check.stateChanged.connect(self.toggle_dem_selection)
        layout.addRow(self.dsm_check)

        self.dem_path = QLineEdit()
        self.dem_path.setPlaceholderText("Select DEM file")
        self.dem_path.setReadOnly(True)
        self.dem_path.setEnabled(False)
        browse_dem_btn = QPushButton("Browse")
        browse_dem_btn.clicked.connect(self.select_dem_file)
        browse_dem_btn.setEnabled(False)
        self.browse_dem_btn = browse_dem_btn
        layout.addRow(QLabel("DEM File:"), self.dem_path)
        layout.addRow("", browse_dem_btn)

        group.setLayout(layout)
        return group

    def browse_kmz(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select KMZ/KML File", "",
            "KMZ/KML Files (*.kmz *.kml)"
        )
        if not file_path:
            return
        if not os.path.exists(file_path):
            QMessageBox.critical(self, "Error", "Selected file does not exist.")
            return
        try:
            self.geometries = extract_geometries_from_kmz(file_path)
            if not self.geometries:
                QMessageBox.critical(self, "Error", "No valid geometries found in file.")
                return
            self.kmz_path.setText(file_path)
            self.geometry_combo.clear()
            self.geometry_combo.addItems([g["name"] for g in self.geometries])
            self.update_selected_geometry(0)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse KMZ/KML: {str(e)}")
            self.kmz_path.clear()
            self.geometry_combo.clear()
            self.area_label.setText("Survey Area: 0.00 acres")

    def update_selected_geometry(self, index):
        if index < 0 or index >= len(self.geometries):
            self.selected_geometry = None
            self.area_label.setText("Survey Area: 0.00 acres")
            self.pattern_score_label.setText("Pattern Score: N/A")
            return
        self.selected_geometry = self.geometries[index]
        try:
            poly = Polygon(self.selected_geometry["coords"])
            area_acres = poly.area * 0.000247105
            self.area_label.setText(f"Survey Area: {area_acres:.2f} acres")
            self.pattern_score = self.selected_geometry.get("pattern_score", 0.0)
            self.pattern_score_label.setText(f"Pattern Score: {self.pattern_score:.2f}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate area: {str(e)}")
            self.area_label.setText("Survey Area: 0.00 acres")

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Export Folder")
        if folder_path:
            if os.access(folder_path, os.W_OK):
                self.export_path.setText(folder_path)
            else:
                QMessageBox.critical(self, "Error", "Selected folder is not writable.")

    def select_dem_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select DEM File", "",
            "GeoTIFF Files (*.tif *.tiff *.dem)"
        )
        if file_path and os.path.exists(file_path):
            self.dem_path.setText(file_path)
        elif file_path:
            QMessageBox.critical(self, "Error", "Selected DEM file does not exist.")

    def toggle_dem_selection(self, state):
        is_checked = state == Qt.CheckState.Checked.value
        self.dem_path.setEnabled(is_checked)
        self.browse_dem_btn.setEnabled(is_checked)
        if not is_checked:
            self.dem_path.clear()

    def update_icon(self, layout_mode):
        self.setWindowIcon(get_app_icon(layout_mode.lower()))

    def export_deliverables(self):
        # Validate inputs
        if not self.kmz_path.text() or not self.selected_geometry:
            QMessageBox.critical(self, "Error", "Please select a valid KMZ/KML file and geometry.")
            return
        if not self.export_path.text() or not os.path.isdir(self.export_path.text()):
            QMessageBox.critical(self, "Error", "Please select a valid export folder.")
            return
        if self.dsm_check.isChecked() and not self.dem_path.text():
            QMessageBox.critical(self, "Error", "Please select a DEM file or disable custom DEM.")
            return
        if self.altitude_spin.value() <= 0 or self.speed_spin.value() <= 0 or self.fov_spin.value() <= 0:
            QMessageBox.critical(self, "Error", "Flight parameters must be positive values.")
            return

        try:
            flight_params = {
                "agl": self.altitude_spin.value() * 0.3048,  # Convert ft to m
                "speed": self.speed_spin.value(),
                "pulse_rate": self.pulse_spin.value(),
                "fov_deg": self.fov_spin.value(),
                "accuracy_mm": self.accuracy_spin.value(),
                "spacing": self.spacing_spin.value()
            }
            result = run_export_pipeline(
                source_kmz_path=self.kmz_path.text(),
                export_folder=self.export_path.text(),
                gcp_spacing=self.spacing_spin.value(),
                layout_mode=self.gcp_layout.currentText().lower(),
                use_dsm=self.dsm_check.isChecked(),
                dem_override_path=self.dem_path.text() if self.dsm_check.isChecked() else None,
                dsm_path=self.dem_path.text() if self.dsm_check.isChecked() else None,
                flight_params=flight_params,
                pass_count=self.pass_count_spin.value()
            )
            self.pattern_score = result.get("Pattern Score", self.pattern_score)
            self.pattern_score_label.setText(f"Pattern Score: {self.pattern_score:.2f}")
            QMessageBox.information(
                self, "Success",
                f"Export completed:\n"
                f"Survey Area: {self.area_label.text().split(': ')[1]}\n"
                f"GCP Layout: {self.gcp_layout.currentText()}\n"
                f"Pattern Score: {self.pattern_score:.2f}\n"
                f"Files saved to: {result['Export Folder']}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def load_stylesheet(self):
        try:
            with open(os.path.join(os.path.dirname(__file__), "styles.qss"), "r") as f:
                self.setStyleSheet(f.read())
        except Exception as e:
            print(f"[ERROR] Failed to load stylesheet: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UASSurveyTool()
    window.setWindowIcon(get_app_icon())
    window.resize(600, 800)
    window.show()
    sys.exit(app.exec())