import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit, QMenu, QFileDialog, QMessageBox
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWebEngineWidgets import QWebEngineView

from maps_functions import generate_leaflet_html, generate_google_maps_html, get_available_displacement_columns


class ProfileMapWidget(QWidget):
    def __init__(self, main_window, bridge_class, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.loaded_run_id = None
        self.loaded_provider = None
        self.loaded_dimension = None

        self.bridge = bridge_class(main_window)
        self.channel = QWebChannel(self)
        self.channel.registerObject("bridge", self.bridge)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)

        self.provider_combo = QComboBox()
        self.provider_combo.addItem("Leaflet / OpenStreetMap", "leaflet")
        self.provider_combo.addItem("Google Maps", "google")
        self.provider_combo.currentIndexChanged.connect(self.on_provider_changed)

        self.api_key_label = QLabel("Google API key:")
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Enter Google Maps API key")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        self.api_key_input.setClearButtonEnabled(True)
        self.api_key_input.setMaximumWidth(320)
        self.api_key_input.setText(self.get_main_window_google_api_key())
        self.api_key_input.returnPressed.connect(self.on_api_key_submitted)

        self.dimension_combo = QComboBox()
        self.dimension_combo.currentIndexChanged.connect(self.on_dimension_changed)

        self.reload_button = QPushButton("Reload map")
        self.reload_button.clicked.connect(lambda: self.refresh(force_reload=True))

        self.export_button = QPushButton("Export")
        self.export_menu = QMenu(self.export_button)
        for label, export_type in (("CSV", "csv"), ("GeoJSON", "geojson"), ("KMZ", "kmz")):
            action = self.export_menu.addAction(label)
            action.triggered.connect(lambda checked=False, value=export_type: self.export_map(value))
        self.export_button.setMenu(self.export_menu)

        toolbar_layout.addWidget(QLabel("Map provider:"))
        toolbar_layout.addWidget(self.provider_combo)
        toolbar_layout.addWidget(self.api_key_label)
        toolbar_layout.addWidget(self.api_key_input)
        toolbar_layout.addWidget(QLabel("Displacement dimension:"))
        toolbar_layout.addWidget(self.dimension_combo)
        toolbar_layout.addWidget(self.reload_button)
        toolbar_layout.addWidget(self.export_button)
        toolbar_layout.addStretch()
        self.update_api_key_visibility("leaflet")

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)

        self.web_view = QWebEngineView()
        self.web_view.page().setWebChannel(self.channel)

        layout.addWidget(toolbar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.web_view, stretch=1)

    def get_run(self):
        return self.main_window.active_run

    def invalidate(self):
        self.loaded_run_id = None
        self.loaded_provider = None
        self.loaded_dimension = None

    def on_provider_changed(self):
        run = self.get_run()
        if run is None:
            return

        run.map_software = self.provider_combo.currentData()
        self.update_api_key_visibility(run.map_software)
        self.refresh(force_reload=True)

    def on_api_key_submitted(self):
        self.save_google_api_key(self.api_key_input.text().strip())
        if self.provider_combo.currentData() == "google":
            self.refresh(force_reload=True)

    def on_dimension_changed(self):
        run = self.get_run()
        displacement_column = self.dimension_combo.currentData()
        if run is None or displacement_column is None:
            return

        run.map_dimension = displacement_column
        self.refresh(force_reload=True)

    def update_dimension_options(self, run):
        columns = get_available_displacement_columns(run.coords)
        preferred = getattr(run, "map_dimension", None)
        if preferred not in columns:
            preferred = columns[0] if columns else None

        self.dimension_combo.blockSignals(True)
        self.dimension_combo.clear()
        for column in columns:
            self.dimension_combo.addItem(f"Dimension {str(column)[4:]}", column)

        if preferred is not None:
            self.dimension_combo.setCurrentIndex(self.dimension_combo.findData(preferred))
            run.map_dimension = preferred

        self.dimension_combo.setEnabled(len(columns) > 1)
        self.dimension_combo.blockSignals(False)
        return preferred

    def get_main_window_google_api_key(self):
        getter = getattr(self.main_window, "get_google_maps_api_key", None)
        if callable(getter):
            return (getter() or "").strip()

        api_input = getattr(self.main_window, "variable_input", {}).get("API")
        return api_input.text().strip() if api_input is not None else ""

    def get_google_maps_api_key(self):
        return self.api_key_input.text().strip() or self.get_main_window_google_api_key()

    def save_google_api_key(self, api_key):
        setter = getattr(self.main_window, "set_google_maps_api_key", None)
        if callable(setter):
            setter(api_key)
            return

        api_input = getattr(self.main_window, "variable_input", {}).get("API")
        if api_input is not None and api_input.text() != api_key:
            api_input.setText(api_key)

    def update_api_key_visibility(self, provider):
        visible = provider == "google"
        self.api_key_label.setVisible(visible)
        self.api_key_input.setVisible(visible)

        if visible and not self.api_key_input.text():
            self.api_key_input.setText(self.get_main_window_google_api_key())

    def get_export_data(self):
        run = self.get_run()
        if run is None or not isinstance(run.coords, pd.DataFrame):
            return pd.DataFrame()
        if not {"LAT", "LONG"}.issubset(run.coords.columns):
            return pd.DataFrame()

        data = run.coords.copy()
        data["LAT"] = pd.to_numeric(data["LAT"], errors="coerce")
        data["LONG"] = pd.to_numeric(data["LONG"], errors="coerce")
        return data.dropna(subset=["LAT", "LONG"])

    @staticmethod
    def export_value(value):
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            value = value.item()
        return value if isinstance(value, (str, int, float, bool)) else str(value)

    def export_map(self, export_type):
        data = self.get_export_data()
        if data.empty:
            QMessageBox.warning(self, "Export Map", "There are no valid map coordinates to export.")
            return

        run = self.get_run()
        base_name = "_".join(str(getattr(run, "name", "map_export")).split()) or "map_export"
        settings = {
            "csv": ("CSV files (*.csv)", ".csv"),
            "geojson": ("GeoJSON files (*.geojson)", ".geojson"),
            "kmz": ("KMZ files (*.kmz)", ".kmz"),
        }
        file_filter, extension = settings[export_type]
        filename, _ = QFileDialog.getSaveFileName(self, f"Export Map as {export_type.upper()}", base_name + extension, file_filter)
        if not filename:
            return

        path = Path(filename)
        if path.suffix.lower() != extension:
            path = path.with_suffix(extension)

        try:
            if export_type == "csv":
                data.to_csv(path, index=False)
            elif export_type == "geojson":
                self.write_geojson(data, path)
            else:
                self.write_kmz(data, path, base_name)
        except Exception as exc:
            QMessageBox.critical(self, "Export Map", f"Could not export the map data: {exc}")
            return

        QMessageBox.information(self, "Export Map", f"Map data exported to:\n{path}")

    def write_geojson(self, data, path):
        features = []
        for _, row in data.iterrows():
            properties = {column: self.export_value(row[column]) for column in data.columns if column not in {"LAT", "LONG"}}
            geometry = {"type": "Point", "coordinates": [float(row["LONG"]), float(row["LAT"])]}
            features.append({"type": "Feature", "geometry": geometry, "properties": properties})

        collection = {"type": "FeatureCollection", "features": features}
        path.write_text(json.dumps(collection, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_kmz(self, data, path, document_name):
        namespace = "http://www.opengis.net/kml/2.2"
        ET.register_namespace("", namespace)
        tag = lambda name: f"{{{namespace}}}{name}"

        kml = ET.Element(tag("kml"))
        document = ET.SubElement(kml, tag("Document"))
        ET.SubElement(document, tag("name")).text = document_name

        for _, row in data.iterrows():
            placemark = ET.SubElement(document, tag("Placemark"))
            ET.SubElement(placemark, tag("name")).text = str(self.export_value(row.get("ID")) or "Profile")
            extended_data = ET.SubElement(placemark, tag("ExtendedData"))

            for column in data.columns:
                value = self.export_value(row[column])
                if value is None:
                    continue
                data_element = ET.SubElement(extended_data, tag("Data"), {"name": str(column)})
                ET.SubElement(data_element, tag("value")).text = str(value)

            point = ET.SubElement(placemark, tag("Point"))
            ET.SubElement(point, tag("coordinates")).text = f'{float(row["LONG"])},{float(row["LAT"])},0'

        kml_bytes = ET.tostring(kml, encoding="utf-8", xml_declaration=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("doc.kml", kml_bytes)

    def refresh(self, force_reload=False):
        run = self.get_run()
        if run is None:
            self.show_message("No active run.")
            return
        if run.status != "complete":
            self.show_message("The map is available after processing finishes.")
            return
        if run.coords is None or run.coords.empty:
            self.show_message("No valid profile coordinates are available.")
            return

        displacement_column = self.update_dimension_options(run)
        if displacement_column is None:
            self.show_message("No displacement dimensions are available.")
            return

        provider = getattr(run, "map_software", "leaflet")
        combo_index = self.provider_combo.findData(provider)
        if combo_index >= 0:
            self.provider_combo.blockSignals(True)
            self.provider_combo.setCurrentIndex(combo_index)
            self.provider_combo.blockSignals(False)
        self.update_api_key_visibility(provider)

        already_loaded = (self.loaded_run_id == run.run_id and self.loaded_provider == provider and self.loaded_dimension == displacement_column)
        if already_loaded and not force_reload:
            return

        try:
            if provider == "google":
                api_key = self.get_google_maps_api_key()
                if not api_key:
                    self.show_message("Enter a Google Maps API key above, then press Enter or click Reload map.")
                    return
                self.save_google_api_key(api_key)
                html = generate_google_maps_html(run.coords, api_key, displacement_column)
            else:
                html = generate_leaflet_html(run.coords, displacement_column)
        except Exception as exc:
            self.show_message(f"Could not create the map: {exc}")
            return

        self.status_label.clear()
        self.web_view.setHtml(html, QUrl("about:blank"))
        self.loaded_run_id = run.run_id
        self.loaded_provider = provider
        self.loaded_dimension = displacement_column

    def show_message(self, message):
        self.status_label.setText(message)
        html = f"""
        <!DOCTYPE html>
        <html>
        <body style="font-family: Arial, sans-serif; display: flex; align-items: center;
                     justify-content: center; height: 100vh; margin: 0;">
            <p>{message}</p>
        </body>
        </html>
        """
        self.web_view.setHtml(html)
