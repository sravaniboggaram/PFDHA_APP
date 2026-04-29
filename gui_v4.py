import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from csv import Sniffer
from re import findall
from matplotlib.pyplot import savefig, subplots, close
from datetime import datetime
import torch
import select_columns
import fit_animation
import create_azimuths
from optimize_v2 import run_optimization
from helper_funs import plot_disp_graph
from pylib_general import gaussian_convolution_nonuniform
from maps_functions import generate_google_maps_html, generate_leaflet_html
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QMenuBar, QMenu, QAction, QFileDialog, QSizePolicy, QLineEdit,
    QComboBox, QRadioButton, QButtonGroup, QDialog, QMessageBox, QTableWidget, 
    QTableWidgetItem, QTabWidget, QScrollArea, QMainWindow, QSlider, QSplitter,
    QApplication, QCheckBox
)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, QTimer, pyqtSlot, QObject, QUrl
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtGui import QColor, QPainter, QBrush, QIcon, QFont
from dataclasses import dataclass


@dataclass(frozen=True)
class EvalConfig:
    strike: str
    parallel: str
    perp: str | None
    ids: str
    az: str
    coord_point1: str
    coord_point2: str
    n_dim: int
    sigma: float
    rand: bool
    uncert: bool
    w_bounds: list | None
    loc_format: bool
    loc_file: str
    prev_ip: bool
    init_p: list | None


class CircleButton(QPushButton):
    def __init__(self):
        super().__init__()
        self.setFixedSize(40, 40)
        #self.setStyleSheet("border: none;")
        self.setIcon(QIcon('caltech_logo.png'))
        self.setIconSize(self.size())

    # def paintEvent(self, event):
    #     painter = QPainter(self)
    #     painter.setRenderHint(QPainter.Antialiasing)
    #     painter.setBrush(QBrush(QColor(255, 140, 0)))  # Orange color
    #     painter.setPen(Qt.black)
    #     painter.drawEllipse(0, 0, self.width(), self.height())


class Bridge(QObject):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

    @pyqtSlot(str, str)
    def show_popup(self, profile_id, file_name):
        self.main_window.show_profile_in_popup(file_name, profile_id)



class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PFDHA")
        self.setMinimumSize(1400, 1100)
        
        # Column Names
        self.strike = 'strike'
        self.parallel = 'parallel'
        self.perp = 'perpendicular'
        self.ids = 'ids'
        self.az = 'az'
        self.coord_point1 = 'point1' # Latitude or Northing
        self.coord_point2 = 'point2' # Longitude or Easting
        self.disp_graph_vals = None
        self.file_names_for_coords = []
        self.get_loc_fun = None

        self.curr_path = None
        self.save_loc = None
        self.file_ext = None
        self.loc_file = None
        self.ind = 0

        self.variable_input = {}

        self.processing_panel = QVBoxLayout()

        self.coords = None
        self.open_windows = []
        self.coords_to_fig = {}
        self.model = None
        self.x_data = None
        self.prof_fig = None
        self.API_KEY = "AIzaSyDI7MgDaJ6pHkLVlZIh48y8T-PbL829988"

        self.file_list = {}

        self.initUI()
        # QTimer.singleShot(1, lambda: self.screen_split.setSizes([1, 1]))
        # QTimer.singleShot(1000, lambda: print("Sizes:", self.screen_split.sizes()))

    def closeEvent(self, event):
        reply = QMessageBox.question(
            self,
            "Exit Confirmation",
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


    def initUI(self):
        # Menu Bar
        menu_bar = QMenuBar()
        file_menu = QMenu("File", self)
        tools_menu = QMenu("Tools", self)
        help_menu = QMenu("Help", self)

        import_file_action = QAction("Save File", self)
        import_file_action.triggered.connect(self.import_file)

        import_folder_action = QAction("Save Folder", self)
        import_folder_action.triggered.connect(self.import_folder)

        file_menu.addAction(import_file_action)
        file_menu.addAction(import_folder_action)

        open_proc_panel_action = QAction("Processing Panel", self)
        open_proc_panel_action.triggered.connect(self.open_proc)
        tools_menu.addAction(open_proc_panel_action)

        map_view_action = QAction("Map View", self)
        map_view_action.triggered.connect(self.open_map_view)
        tools_menu.addAction(map_view_action)

        fitting = QAction("Adjust Fit", self)
        fitting.triggered.connect(self.open_fitting_window)
        tools_menu.addAction(fitting)

        self.disp_graph_action = QAction("Displacement Graph", self)
        self.disp_graph_action.setEnabled(False)
        self.disp_graph_action.triggered.connect(self.display_disp_graph)
        tools_menu.addAction(self.disp_graph_action)

        menu_bar.addMenu(file_menu)
        menu_bar.addMenu(tools_menu)
        menu_bar.addMenu(help_menu)

        circle_btn = CircleButton()

        # Top Layout
        top_widget = QWidget()
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)
        top_layout.addWidget(circle_btn)
        top_layout.addWidget(menu_bar)
        top_layout.addStretch()
        top_widget.setLayout(top_layout)
        top_widget.setFixedHeight(menu_bar.sizeHint().height() + 10)

        self.create_proc_panel_widgets()

        self.screen_split = QSplitter(Qt.Horizontal)
        
        self.left_frame = QFrame()
        self.left_frame.setFrameShape(QFrame.Box)
        self.left_frame.setLayout(self.processing_panel)
        #self.left_frame.setFixedWidth(450)
        self.left_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.screen_split.addWidget(self.left_frame)
        
        # Main Area
        self.main_area = QFrame()
        self.main_area.setFrameShape(QFrame.Box)
        self.main_area.setStyleSheet("background-color: #f0f0f0;")
        self.main_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_area_layout = QVBoxLayout()
        self.main_area.setLayout(self.main_area_layout)
        self.screen_split.addWidget(self.main_area)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        #self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.scroll_content = QWidget()
        self.inner_layout = QVBoxLayout(self.scroll_content)
        self.inner_layout.setAlignment(Qt.AlignTop)
        self.inner_layout.setSpacing(8)
        self.inner_layout.setContentsMargins(5, 5, 5, 5)

        self.scroll_area.setWidget(self.scroll_content)

        # Left Panel
        self.open_proc()

        # Final Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(top_widget)
        main_layout.addWidget(self.screen_split)
        self.setLayout(main_layout)
        self.screen_split.setSizes([300,900])


    # def open_map_view(self):
    #     map_window = QMainWindow()
    #     map_window.setWindowTitle("Map View")
    #     map_window.resize(800, 600)

    #     central_widget = QWidget()
    #     layout = QVBoxLayout(central_widget)
    #     map_window.setCentralWidget(central_widget)

    #     #html = self.generate_leaflet_html(self.coords)
    #     html = self.generate_google_maps_html(self.coords)

    #     web_view = QWebEngineView()
    #     web_view.setHtml(html)

    #     layout.addWidget(web_view)
    #     map_window.show()
    #     self.open_windows.append(map_window)
    #     map_window.destroyed.connect(lambda: self.open_windows.remove(map_window))

    
    # def generate_google_maps_html(self, coords_df):
    #     if coords_df is None or coords_df.empty:
    #         return "<html><body><p>No coordinates provided.</p></body></html>"

    #     api_key = self.API_KEY
    #     first_row = coords_df.iloc[0]
    #     center_lat = first_row['latitude']
    #     center_lon = first_row['longitude']

    #     marker_js = ""
    #     list_items_html = ""

    #     i = 0

    #     for index, row in coords_df.iterrows():
    #         lat = row['latitude']
    #         lon = row['longitude']
    #         profile_id = row['Profile ID']
    #         disp = row['rescaled disp']
    #         width = row['actual width']
    #         file_name = row['file_name']
            

    #         marker_js += f"""
    #             const marker{index} = new google.maps.Marker({{
    #                 position: {{ lat: {lat}, lng: {lon} }},
    #                 map: map,
    #                 title: "Profile ID: {profile_id}",
    #                 label: {{
    #                     text: "{profile_id}",
    #                     color: "black",
    #                     fontWeight: "bold"
    #                 }}
    #             }});
    #             bounds.extend(marker{index}.getPosition());
    #             markerMap["{profile_id}"] = marker{index}.getPosition();
    #         """

    #         list_items_html += f"""
    #             <div class="profile-row" onclick="focusOnMarker('{profile_id}')">
    #                 <strong>{profile_id}</strong><br>
    #                 Rescaled Disp: {np.round([disp], 3)}<br>
    #                 Width: {np.round([width], 3)}
    #             </div>
    #         """

    #     html = f"""
    #     <!DOCTYPE html>
    #     <html>
    #     <head>
    #         <meta charset="utf-8" />
    #         <title>Map View</title>
    #         <meta name="viewport" content="width=device-width, initial-scale=1.0">
    #         <style>
    #             html, body {{
    #                 height: 100%;
    #                 margin: 0;
    #                 font-family: Arial, sans-serif;
    #             }}
    #             #container {{
    #                 display: flex;
    #                 height: 100%;
    #                 width: 100%;
    #             }}
    #             #sidebar {{
    #                 width: 300px;
    #                 overflow-y: auto;
    #                 background: #f8f8f8;
    #                 border-right: 1px solid #ccc;
    #                 padding: 10px;
    #                 box-sizing: border-box;
    #             }}
    #             #map {{
    #                 flex-grow: 1;
    #             }}
    #             .profile-row {{
    #                 padding: 10px;
    #                 margin-bottom: 8px;
    #                 background: #fff;
    #                 border: 1px solid #ddd;
    #                 border-radius: 5px;
    #                 cursor: pointer;
    #                 transition: background 0.3s;
    #             }}
    #             .profile-row:hover {{
    #                 background: #eef;
    #             }}
    #         </style>
    #     </head>
    #     <body>
    #         <div id="container">
    #             <div id="sidebar">
    #                 <h3>Profiles</h3>
    #                 {list_items_html}
    #             </div>
    #             <div id="map"></div>
    #         </div>

    #         <script>
    #             let map;
    #             const markerMap = {{}};

    #             function loadScript(src) {{
    #                 return new Promise((resolve, reject) => {{
    #                     const script = document.createElement('script');
    #                     script.src = src;
    #                     script.async = true;
    #                     script.defer = true;
    #                     script.onload = resolve;
    #                     script.onerror = reject;
    #                     document.head.appendChild(script);
    #                 }});
    #             }}

    #             async function initMap() {{
    #                 await loadScript("https://maps.googleapis.com/maps/api/js?key={api_key}&libraries=marker");

    #                 const center = {{ lat: {center_lat}, lng: {center_lon} }};
    #                 map = new google.maps.Map(document.getElementById('map'), {{
    #                     center: center,
    #                     zoom: 13
    #                 }});
    #                 const bounds = new google.maps.LatLngBounds();

    #                 {marker_js}

    #                 if (Object.keys(markerMap).length > 1) {{
    #                     map.fitBounds(bounds);
    #                 }}
    #             }}

    #             function focusOnMarker(profileId) {{
    #                 const position = markerMap[profileId];
    #                 if (position) {{
    #                     map.panTo(position);
    #                     map.setZoom(15);
    #                 }}
    #             }}

    #             window.onload = initMap;
    #         </script>
    #     </body>
    #     </html>
    #     """
    #     return html


    def open_map_view(self):

        map_sftwr = self.variable_input['map_sftwr'].checkedId()

        map_window = QMainWindow()
        map_window.setWindowTitle("Map View")
        map_window.resize(800, 600)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        map_window.setCentralWidget(central_widget)

        web_view = QWebEngineView()
        
        if map_sftwr == 0:
            channel = QWebChannel()
            bridge = Bridge(self)
            channel.registerObject("bridge", bridge)
            web_view.page().setWebChannel(channel)

            html_path = generate_google_maps_html(self.coords, self.API_KEY)
        else:
            html_path = generate_leaflet_html(self.coords)

        #web_view.setHtml(html)
        abs_path = Path(html_path).resolve()
        web_view.load(QUrl.fromLocalFile(str(abs_path)))
        layout.addWidget(web_view)

        map_window.show()
        self.open_windows.append(map_window)
        map_window.destroyed.connect(lambda: self.open_windows.remove(map_window))

    

    def open_fitting_window(self):

        fitting_window = QMainWindow()
        fitting_window.setWindowTitle("Adjust Parameters")
        fitting_window.resize(1400, 1200)

        splitter = QSplitter(Qt.Horizontal, fitting_window)
        fitting_window.setCentralWidget(splitter)

        slider_panel = QWidget()
        slider_layout = QVBoxLayout()
        slider_panel.setLayout(slider_layout)
        splitter.addWidget(slider_panel)

        fig_disp = Figure(figsize=(6, 4))
        canvas = FigureCanvas(fig_disp)
        ax = fig_disp.add_subplot(111)
        splitter.addWidget(canvas)
        splitter.setSizes([3,2])

        x_tensor = torch.tensor(self.x_data, dtype=float).unsqueeze(0).T

        self.prof_fig.canvas.draw()
        w, h = self.prof_fig.canvas.get_width_height()
        buf = np.frombuffer(self.prof_fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(h, w, 4)[:, :, :3]
        #h, w, _ = buf.shape

        bckgnd = buf[:, w//2:, :]


        def update_figure():
            ax.clear()

            ax.imshow(bckgnd, aspect='auto', 
                      extent=[min(self.x_data), max(self.x_data), min(self.x_data), max(self.x_data)], 
                      alpha=0.5)

            y_pred = self.model(x_tensor).detach().numpy()

            ax.plot(self.x_data, y_pred, color="steelblue", label="Model output")
            ax.legend()
            ax.set_title("Interactive Model Fit")
            ax.grid(True)
            canvas.draw_idle()

        def update_model(name, label, slider_val):
            new_value = slider_val / 10.0
            label.setText(f"{name}: {new_value:.2f}")

            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if n == name:
                        p.copy_(torch.tensor(new_value))
                        break

            update_figure()

        for name, param in self.model.named_parameters():

            init_val = float(param.item())

            label = QLabel(f'{name}: {init_val:.2f}')
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-100, 100)
            slider.setValue(int(init_val * 10))

            slider.valueChanged.connect(
                lambda val, n=name, l=label: update_model(n, l, val)
            )

            slider_layout.addWidget(label)
            slider_layout.addWidget(slider)

            setattr(self, f"{name}_slider", slider)
            setattr(self, f"{name}_label", label)

        update_figure()

        fitting_window.show()

        self.open_windows.append(fitting_window)
        fitting_window.destroyed.connect(lambda: self.open_windows.remove(fitting_window))
    

    def display_disp_graph(self):
        disp_fig = plot_disp_graph(self.disp_graph_vals)
        canvas_disp = FigureCanvas(disp_fig)

        graph_window = QMainWindow()
        graph_window.setWindowTitle("Displacement Graph")
        graph_window.resize(1400, 1200)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(canvas_disp)

        graph_window.setCentralWidget(central_widget)
        graph_window.show()

        self.open_windows.append(graph_window)
        graph_window.destroyed.connect(lambda: self.open_windows.remove(graph_window))

    def create_proc_panel_widgets(self):

        self.proc_panel_title = QLabel("STARTUP")
        self.proc_panel_title.setStyleSheet("font-size: 20px; font-weight: bold")
        self.title_line = self._hline()

        self.proc_panel_stack = QWidget()
        self.proc_panel_stack_layout = QVBoxLayout()
        self.proc_panel_stack_layout.setContentsMargins(0, 0, 0, 0)
        self.proc_panel_stack.setLayout(self.proc_panel_stack_layout)

        self.next_back_layout = QHBoxLayout()
        self.proc_panel_next = QPushButton("Next")
        self.proc_panel_next.clicked.connect(lambda: self.proc_next_back(1))
        self.proc_panel_back = QPushButton("Back")
        self.proc_panel_back.clicked.connect(lambda: self.proc_next_back(-1))

        ## DATA PANEL WIDGETS ##
        data_title = QLabel("Data")
        data_title.setStyleSheet('font-weight: bold; font-size: 22px;')
        save_title = QLabel("Saved To")
        save_title.setStyleSheet('font-weight: bold; font-size: 22px;')
        file_button = QPushButton("Import Data File")
        file_button.clicked.connect(self.import_file)
        folder_button = QPushButton("Import Data Folder")
        folder_button.clicked.connect(self.import_folder)
        save_location_button = QPushButton("Save Location")
        save_location_button.clicked.connect(lambda: self.import_folder(True))
        self.import_label = QLabel("No Data Imported")
        self.import_label.setStyleSheet("color: gray;")
        self.import_label.setWordWrap(True)
        self.save_label = QLabel("No Folder Selected")
        self.save_label.setStyleSheet("color: gray;")
        self.save_label.setWordWrap(True)

        data_widgets = [data_title, self.import_label, file_button, folder_button,
                        save_title, self.save_label, save_location_button]
        
        ## SET PARAMETERS WIDGETS ##
        params_title = QLabel("Set Parameters")
        params_title.setStyleSheet('font-weight: bold; font-size: 22px;')
        # Peak smoothing sigma
        peak_sigma_label = QLabel("Set \u03C3 for smoothing")
        peak_sigma_label.setWordWrap(True)
        peak_sigma_line_edit = QLineEdit()
        peak_sigma_line_edit.setPlaceholderText("Enter Value")
        self.variable_input['sigma'] = peak_sigma_line_edit
        # Optimization iter
        epochs_label = QLabel("Set max epoch count")
        epochs_label.setWordWrap(True)
        epoch_line_edit = QLineEdit()
        epoch_line_edit.setPlaceholderText("Enter Value")
        self.variable_input['n_iter'] = epoch_line_edit
        # Number of dimensions dropdown
        dim_label = QLabel("Select number of dimensions")
        combo_dim = QComboBox()
        combo_dim.addItems(['1', '2', '3'])
        self.variable_input['n_dim'] = combo_dim

        # Interpolate Data
        interp_label = QLabel("Create interpolated data")
        interp_n = QLineEdit()
        interp_n.setPlaceholderText("Enter total # data points")
        self.variable_input['interp'] = interp_n

        params_widgets = [params_title, peak_sigma_label, peak_sigma_line_edit,
                               epochs_label, epoch_line_edit, dim_label, combo_dim,
                               interp_label, interp_n]


        ## MODES WIDGETS ##
        modes_title = QLabel("Choose Modes")
        modes_title.setStyleSheet('font-weight: bold; font-size: 22px;')

        ip_title = QLabel("Initial Parameters")

        # Standard IP
        rb_stdip_layout = QHBoxLayout()
        radio_stdip = QRadioButton()
        label_stdip = QLabel("Standard Initial Parameters")
        #label1.setWordWrap(True)
        label_stdip.setStyleSheet("margin-left: 5px;")
        info_stdip = QLabel("ⓘ")
        info_stdip.setToolTip("Use a predetermined set of IP based on the provided profile")
        info_stdip.setStyleSheet("color: black; margin-left: 5px;")

        # Random IP
        rb_rndip_layout = QHBoxLayout()
        radio_rndip = QRadioButton()
        label_rndip = QLabel("Random Initial Parameters")
        #label2.setWordWrap(True)
        label_rndip.setStyleSheet("margin-left: 5px;")
        info_rndip = QLabel("ⓘ")
        info_rndip.setToolTip("Use 15 sets of randomly chosen IP and select best fit")
        info_rndip.setStyleSheet("color: black; margin-left: 5px;")

        # IP Button Group
        ip_group = QButtonGroup(self)
        ip_group.addButton(radio_stdip, id=0)
        ip_group.addButton(radio_rndip, id=1)
        radio_stdip.setChecked(True)

        prev_ip_layout = QHBoxLayout()
        prev_ip_chckbox = QCheckBox()
        prev_ip_chckbox.setChecked(False)
        label_previp = QLabel("Use Previous Profile Parameters")
        label_previp.setStyleSheet("margin-left: 5px;")
        info_previp = QLabel("ⓘ")
        info_previp.setToolTip("Use the previous profile's final parameters as the initial parmeters for the next profile")
        info_previp.setWordWrap(True)
        info_previp.setStyleSheet("color: black; margin-left: 5px;")
        self.variable_input['prev_ip'] = prev_ip_chckbox

        output_title = QLabel("Output Data")

        # Mean
        rb_mean_layout = QHBoxLayout()
        radio_mean = QRadioButton()
        label_mean = QLabel("Mean Only")
        #label3.setWordWrap(True)
        label_mean.setStyleSheet("margin-left: 5px;")
        info_mean = QLabel("ⓘ")
        info_mean.setToolTip("Only calculate parameter values")
        info_mean.setStyleSheet("color: black; margin-left: 5px;")

        # Mean and Uncertainty
        rb_munc_layout = QHBoxLayout()
        radio_munc = QRadioButton()
        label_munc = QLabel("Mean and Uncertainties")
        #label4.setWordWrap(True)
        label_munc.setStyleSheet("margin-left: 5px;")
        info4_munc = QLabel("ⓘ")
        info4_munc.setToolTip("Calculates both parameters and offsets")
        info4_munc.setStyleSheet("color: black; margin-left: 5px;")

        # Mean and Uncertainty Button Group
        mode_group = QButtonGroup(self)
        mode_group.addButton(radio_mean, id=0)
        mode_group.addButton(radio_munc, id=1)
        radio_mean.setChecked(True)

        window_size_label = QLabel("Uncertainty Calculation Window Size")
        window_min = QLineEdit()
        window_min.setPlaceholderText("Enter Minimum Window Size")
        self.variable_input['w_min'] = window_min
        window_max = QLineEdit()
        window_max.setPlaceholderText("Enter Maximum Window Size")
        self.variable_input['w_max'] = window_max

        self.variable_input['IP'] = ip_group
        self.variable_input['Mode'] = mode_group

        modes_widgets = [modes_title, ip_title, {rb_stdip_layout: [radio_stdip, label_stdip, info_stdip]}, 
                              {rb_rndip_layout: [radio_rndip, label_rndip, info_rndip]}, 
                              {prev_ip_layout: [prev_ip_chckbox, label_previp, info_previp]},
                              output_title, {rb_mean_layout: [radio_mean, label_mean, info_mean]}, 
                              {rb_munc_layout: [radio_munc, label_munc, info4_munc]}, window_size_label,
                              window_min, window_max]
        
        ## LOCATION WIDGETS ##
        loc_title = QLabel("Calculate Center Locations")
        loc_title.setStyleSheet('font-weight: bold; font-size: 22px;')

        format_title = QLabel("Location Format")

        # rb_nonefmt_layout = QHBoxLayout()
        # radio_nonefmt = QRadioButton()
        # label_nonefmt = QLabel("None")
        # label_nonefmt.setWordWrap(True)
        # label_nonefmt.setStyleSheet("margin-left: 5px;")

        def enable_loc_format():
            if not radio_nonefl.isChecked():
                radio_utm.setDisabled(False)
                radio_latlon.setDisabled(False)
            else:
                radio_utm.setDisabled(True)
                radio_latlon.setDisabled(True)

        rb_utm_layout = QHBoxLayout()
        radio_utm = QRadioButton()
        radio_utm.setDisabled(True)
        label_utm = QLabel("UTM")
        label_utm.setWordWrap(True)
        label_utm.setStyleSheet("margin-left: 5px;")
        info_utm = QLabel("ⓘ")
        info_utm.setToolTip("Provide profile's origin Easting, Northing, Zone Number, Zone Letter")
        info_utm.setWordWrap(True)
        info_utm.setStyleSheet("color: black; margin-left: 5px;")

        rb_latlon_layout = QHBoxLayout()
        radio_latlon = QRadioButton()
        radio_latlon.setDisabled(True)
        label_latlon = QLabel("LATLON")
        label_latlon.setWordWrap(True)
        label_latlon.setStyleSheet("margin-left: 5px;")
        info_latlon = QLabel("ⓘ")
        info_latlon.setToolTip("Provide profile's origin Latitude, Longitude, Azimuth")
        info_latlon.setWordWrap(True)
        info_latlon.setStyleSheet("color: black; margin-left: 5px;")

        loc_format_group = QButtonGroup(self)
        #loc_format_group.addButton(radio_nonefmt, id=-2)
        loc_format_group.addButton(radio_utm, id=0)
        loc_format_group.addButton(radio_latlon, id=1)
        radio_latlon.setChecked(True)

        input_type_title = QLabel("Location Input Type")

        rb_nonefl_layout = QHBoxLayout()
        radio_nonefl = QRadioButton()
        label_nonefl = QLabel("None")
        label_nonefl.setWordWrap(True)
        label_nonefl.setStyleSheet("margin-left: 5px;")

        rb_infl_layout = QHBoxLayout()
        radio_infl = QRadioButton()
        label_infl = QLabel("In-file")
        label_infl.setWordWrap(True)
        label_infl.setStyleSheet("margin-left: 5px;")
        info_infl = QLabel("ⓘ")
        info_infl.setToolTip("Coordinate data will be provided as a part of the data file, for ex: as columns")
        info_infl.setWordWrap(True)
        info_infl.setStyleSheet("color: black; margin-left: 5px;")

        rb_sepfl_layout = QHBoxLayout()
        radio_sepfl = QRadioButton()
        label_sepfl = QLabel("Separate File")
        label_sepfl.setWordWrap(True)
        label_sepfl.setStyleSheet("margin-left: 5px;")
        info_sepfl = QLabel("ⓘ")
        info_sepfl.setToolTip("Coordinate data will be provided separate from the data file")
        info_sepfl.setWordWrap(True)
        info_sepfl.setStyleSheet("color: black; margin-left: 5px;")

        def on_file_group_changed(id):
            loc_file_button.setEnabled(id == 2)

        file_group = QButtonGroup(self)
        file_group.addButton(radio_nonefl, id=0)
        file_group.addButton(radio_infl, id=1)
        file_group.addButton(radio_sepfl, id=2)
        radio_nonefl.setChecked(True)
        file_group.buttonClicked[int].connect(on_file_group_changed)
        file_group.buttonClicked.connect(enable_loc_format)

        loc_file_button = QPushButton("Location Coordinates File")
        loc_file_button.clicked.connect(lambda: self.import_file(True))
        loc_file_button.setEnabled(False)

        self.variable_input['loc_format'] = loc_format_group
        self.variable_input['loc_inp'] = file_group

        map_sftwr_title = QLabel("Map Software")

        rb_gglmap_layout = QHBoxLayout()
        radio_gglmap = QRadioButton()
        label_gglmap = QLabel("Google Maps")
        label_gglmap.setWordWrap(True)
        label_gglmap.setStyleSheet("margin-left: 5px;")
        info_gglmap = QLabel("ⓘ")
        info_gglmap.setToolTip("Follow instructions in Help Menu to get your Google Maps API key")
        info_gglmap.setWordWrap(True)
        info_gglmap.setStyleSheet("color: black; margin-left: 5px;")

        api_key_line_edit = QLineEdit()
        api_key_line_edit.setPlaceholderText("Enter Google Maps API Key")
        api_key_line_edit.setEnabled(False)
        self.variable_input['API'] = api_key_line_edit

        rb_lfltmap_layout = QHBoxLayout()
        radio_lfltmap = QRadioButton()
        label_lfltmap = QLabel("Leaflet Maps")
        label_lfltmap.setWordWrap(True)
        label_lfltmap.setStyleSheet("margin-left: 5px;")
        info_lfltmap = QLabel("ⓘ")
        info_lfltmap.setToolTip("Free map software")
        info_lfltmap.setWordWrap(True)
        info_lfltmap.setStyleSheet("color: black; margin-left: 5px;")

        def map_sftwr_group_changed(id):
            api_key_line_edit.setEnabled(id == 0)

        map_group = QButtonGroup(self)
        map_group.addButton(radio_gglmap, id=0)
        map_group.addButton(radio_lfltmap, id=1)
        radio_lfltmap.setChecked(True)
        map_group.buttonClicked[int].connect(map_sftwr_group_changed)

        self.variable_input['map_sftwr'] = map_group

        self.start_button = QPushButton("START")
        self.start_button.clicked.connect(self.process)

        loc_widgets = [loc_title, input_type_title, {rb_nonefl_layout: [radio_nonefl, label_nonefl]}, 
                       {rb_infl_layout: [radio_infl, label_infl, info_infl]}, 
                        {rb_sepfl_layout: [radio_sepfl, label_sepfl, info_sepfl]},
                        format_title, #{rb_nonefmt_layout: [radio_nonefmt, label_nonefmt]}, 
                        {rb_utm_layout: [radio_utm, label_utm, info_utm]}, 
                        {rb_latlon_layout: [radio_latlon, label_latlon, info_latlon]},
                        loc_file_button, map_sftwr_title, 
                        {rb_gglmap_layout: [radio_gglmap, label_gglmap, info_gglmap]}, 
                        api_key_line_edit, 
                        {rb_lfltmap_layout: [radio_lfltmap, label_lfltmap, info_lfltmap]}, 
                        self.start_button]
        
        self.proc_panel_widgets = [data_widgets, params_widgets, modes_widgets,
                                   loc_widgets]
     

    def display_proc_widgets(self):
        self.clear_layout(self.proc_panel_stack_layout)
        self.show_in_layout(self.proc_panel_stack_layout, self.proc_panel_widgets[self.ind])
        self.start_button.setEnabled(self.curr_path != None)
        

    def proc_next_back(self, dir):        
        self.ind += dir
        self.ind = max(0, min(self.ind, len(self.proc_panel_widgets) - 1))
        if self.ind == 0:
            self.proc_panel_back.setParent(None)
        elif self.ind == len(self.proc_panel_widgets) - 1:
            self.proc_panel_next.setParent(None)
        else:
            self.next_back_layout.addWidget(self.proc_panel_back, alignment=Qt.AlignLeft)
            self.next_back_layout.addWidget(self.proc_panel_next, alignment=Qt.AlignRight)
        self.display_proc_widgets()

    
    def import_file(self, loc_file=False):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File")

        if not file_path:
            return
        
        if loc_file:
            self.loc_file = Path(file_path)
        else:
            self.import_label.setText(file_path)
            self.curr_path = Path(file_path)

    def import_folder(self, save=False):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            if save:
                self.save_loc = folder_path
                self.save_label.setText(self.save_loc)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                self.fig_dir = Path(self.save_loc) / f"figures_{timestamp}"
                self.fig_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.import_label.setText(folder_path)
                self.curr_path = Path(folder_path)

    def _hline(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: #aaa;")
        return line

    def open_proc(self):
        if self.proc_panel_title.text() != "PROCESSING PANEL":
            self.clear_layout(self.processing_panel)
            self.ind = 0

            self.proc_panel_title.setText("PROCESSING PANEL")
            self.processing_panel.addWidget(self.proc_panel_title)
            self.processing_panel.addWidget(self.title_line)
            self.processing_panel.addWidget(self.proc_panel_stack)

            #self.display_choose_data()
            self.display_proc_widgets()
            self.processing_panel.addStretch()
            
            
            self.next_back_layout.addWidget(self.proc_panel_next, alignment=Qt.AlignRight)
            self.processing_panel.addLayout(self.next_back_layout)
        

    def check_inputs_entered(self, button):
        all_valid = True
        if self.curr_path == None:
            all_valid = False

        button.setEnabled(all_valid)


    def clear_layout(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()

            if widget:
                layout.removeWidget(widget)
                widget.setParent(None)
            elif child_layout:
                self.clear_layout(child_layout)
                layout.removeItem(child_layout)
                child_layout.setParent(None)


    def show_in_layout(self, layout, items):
        for item in items:
            if isinstance(item, QWidget):
                # if item.parent() is None:
                #     item.setParent(layout.parentWidget())
                layout.addWidget(item)
            elif isinstance(item, dict):
                child_layout = next(iter(item))
                self.show_in_layout(child_layout, item[child_layout])
                layout.addLayout(child_layout)
        layout.addStretch()

    def set_loc_format(self):
        loc_format_map = {0: 'UTM', 1: 'LATLON'}
        if self.variable_input['loc_inp'].checkedId() != 0:
            self.loc_format = loc_format_map[self.variable_input['loc_format'].checkedId()]
            (self.coord_point1, self.coord_point2) = ('easting', 'northing') if self.loc_format == 'UTM' else ('lat', 'lon')
        else:
            self.loc_format = None

        if self.loc_file:
            if self.loc_file.suffix == '.geojson':
                self.loc_file = create_azimuths.create_utm_loc_file_from_geojson(self.loc_file)
            #names = [self.ids, self.az, self.coord_point1, self.coord_point2] if self.loc_format == 'LATLON' else [self.ids, self.az, self.coord_point1, self.coord_point2, 'ZL', 'ZN']
            self.loc_file = pd.read_csv(self.loc_file)


    def process(self):
        self.n_dim = int(self.variable_input['n_dim'].currentText())
        #fix
        self.sigma = float(self.variable_input['sigma'].text()) if len(self.variable_input['sigma'].text())>0 else 10
        self.set_loc_format()
        self.rand = bool(self.variable_input['IP'].checkedId())
        self.uncert = bool(self.variable_input['Mode'].checkedId())
        w_min = float(self.variable_input['w_min'].text()) if len(self.variable_input['w_min'].text())>0 else None
        w_max = float(self.variable_input['w_max'].text()) if len(self.variable_input['w_max'].text())>0 else None
        self.w_bounds = None if w_min is None and w_max is None else [w_min, w_max]
        self.prev_ip = self.variable_input['prev_ip'].isChecked()
        self.init_p = None

        if self.curr_path.suffix == '.h5':
            self.read_h5()
        else:
            self.read_txt()

    # def get_in_file_loc(self, name, df):
        
    #     coords = None
    #     lat, lon, az = df.iloc[0][self.lat], df.iloc[0][self.lon], df.iloc[0][self.az]
    #     if pd.notna(lat) and pd.notna(lon) and pd.notna(az):
    #         coords = [lat, lon, az]
    #         self.file_names_for_coords.append(name)

    #     return coords


    # def get_sep_file_loc(self, id):
    #     row = self.location_file.loc([self.location_file]['ID'] == id)
    #     coords = 


    def get_txt_cols(self, file):       

        line = open(file, 'r').readline()
        if any(c.isalpha() for c in line):
            header = 0
        else:
            header = None
    
        delim = Sniffer().sniff(line).delimiter
        data = pd.read_csv(file, delimiter=delim, header=header)
        data.dropna()

        if header == None:
            data.columns = [str(name) for name in data.columns]

        if data.ndim == 1:
            n_col = 1
            #add error message

        dialog = select_columns.SelectColumns(data.columns, self.n_dim, self.variable_input['loc_inp'].checkedId() == 1, self.loc_format)

        if dialog.exec_() == QDialog.Accepted:
            option_strike, option_parallel, option_perpendicular, prof_id, point1, point2, az = dialog.selected_options()

            options_text = f"<b>Distance Along Strike:</b> {option_strike}\n<b>Parallel Displacement:</b> {option_parallel}"
            if option_perpendicular:
                options_text += f"\n<b>Perpendicular Displacement:</b> {option_perpendicular}"
            (text1, text2) = ("<b>Latitude:</b>", "<b>Longitude:</b>") if self.loc_format == 'LATLON' else ("<b>Northing:</b>", "<b>Easting:</b>")
            options_text += f"\n<b>Profile ID:</b> {prof_id} \n {text1} {point1} \n {text2} {point2} \n<b>Azimuth:</b> {az}"

            msg = QMessageBox()
            msg.setFixedSize(400, 400)
            msg.setTextFormat(Qt.RichText)
            msg.information(self, "Columns Selected", options_text)
            
            cols = [option_strike, option_parallel]
            new_names = {option_strike: self.strike, option_parallel: self.parallel}
            
            if option_perpendicular:
                cols.append(option_perpendicular)
                new_names[option_perpendicular] = self.perp

            if prof_id != "None":
                cols.append(prof_id)
                new_names[prof_id] = self.ids

            if point1 != None and point2 != None and az != None:
                cols.extend([point1, point2, az])
                new_names[point1] = self.coord_point1
                new_names[point2] = self.coord_point2
                new_names[az] = self.az

            if header == None:
                for i in range(len(cols)):
                    new_names[int(cols[i])] = new_names[cols[i]]
                    del new_names[cols[i]]
                    cols[i] = int(cols[i])
            
            return cols, new_names, prof_id, delim, header
        return None
    
    def get_path_name(self, path):
        path = str(path).replace('\\', '/')
        ind1 = path.rfind("/") + 1
        ind2 = path.rfind(".")
        nums = findall(r'\d+', path[ind1:ind2])
        return path[ind1:ind2], ''.join(nums)
    
    def process_locations(self, row, file_num):
        if self.loc_file is not None:
            row = self.loc_file[self.loc_file[self.ids] == file_num]

        coords = [row[self.coord_point1], row[self.coord_point2], row[self.az]] # E,N,A or Lat, Lon, A
            
        if self.loc_format == 'UTM':
            coords.extend([row['ZL'], row['ZN']])

        return coords if pd.notna(coords).all() else None
    

    def eval_data(self, df, file_num, file_info=None, folder_name=None, file_name=None, history=False):
        coords = None
        # FIX SIGMA_X!!!
        y1 = gaussian_convolution_nonuniform(df[self.strike], df[self.parallel], sigma_x=5)
        smooth_data = np.vstack([df[self.strike], y1]).T

        if self.n_dim > 1:
            y2 = gaussian_convolution_nonuniform(df[self.strike], df[self.perp], sigma_x=5)
            smooth_data = np.hstack([smooth_data, y2[np.newaxis, :].T])

        df_data = df[[self.strike, self.parallel]] if self.n_dim < 2 else df[[self.strike, self.parallel, self.perp]]
        
        if history:
            return run_optimization(smooth_data, df_data.to_numpy(), file_num, self.sigma, self.rand, history=history)

        if self.loc_format:
            coords = self.process_locations(df.iloc[0], file_num)
            if coords:
                self.file_names_for_coords.append(folder_name or file_name)
        
        table, fig, model, u, losses, init_p = run_optimization(smooth_data, df_data.to_numpy(), file_num, self.sigma, self.rand, self.uncert, coords, self.w_bounds)
        self.init_p = init_p if self.prev_ip else None
        self.file_list.setdefault(folder_name if folder_name else file_name, {})[file_num] = (fig, table, 
                                                                                              (model, df_data[self.strike].iloc[0], df_data[self.strike].iloc[-1], len(df_data[self.strike])), 
                                                                                              u, losses, file_info)
        self.show_file_selector()

        if self.save_loc:
            savefig(self.fig_dir / f"{file_name}_profile{file_num}.png")
            close(fig)

        return table


    def read_txt(self):

        is_file = self.curr_path.is_file()
        folder = [self.curr_path] if is_file else list(self.curr_path.glob("*"))

        (folder_name, _) = self.get_path_name(self.curr_path) if not is_file else (None, None)
        test_file = self.curr_path if is_file else next(self.curr_path.iterdir())

        cols_data = self.get_txt_cols(test_file)
        if not cols_data:
            return
        
        self.cols, self.new_names, prof_id, self.delim, self.header = cols_data[0], cols_data[1], cols_data[2], cols_data[3], cols_data[4]
        csv_fold = pd.DataFrame()
        csv_fold_file = ""
        i = 0

        for file in folder:
            file_name, file_num = self.get_path_name(file)
            if file_num == '':
                file_num = i
            data = pd.read_csv(file, delimiter=self.delim, header=self.header)
            i += 1
            df = data[self.cols].rename(columns=self.new_names)

            if prof_id == "None":
                # Single Profile
                table = self.eval_data(df, file_num, (file, None), folder_name, file_name)
                csv_fold = pd.concat([csv_fold, table], ignore_index=True)
            else:
                # Multiple Profiles in 1 file
                profiles = df.groupby(self.ids)
                profile_ids = list(profiles.groups.keys())[:2]
                csv = pd.DataFrame()
                csv_fold = None

                for id in profile_ids:
                    orig_prof = profiles.get_group(id)
                    table = self.eval_data(orig_prof, id, (file, id), folder_name, file_name)
                    csv = pd.concat([csv, table], ignore_index=True)

                self.disp_graph_vals = csv['rescaled disp']
                self.disp_graph_action.setEnabled(True)
                
                if self.save_loc:
                    csv.to_excel(self.save_loc + "/" + file_name + ".xlsx")
                
                #self.coords = np.vstack([csv['latitude'].dropna().tolist(), csv['longitude'].dropna().tolist()]).T
                self.coords = csv[['latitude', 'longitude', 'Profile ID', 'rescaled disp', 'actual width']].dropna().reset_index(drop=True)
            
                self.coords['file_name'] = pd.Series(self.file_names_for_coords)

        if isinstance(csv_fold, pd.DataFrame):
            self.disp_graph_vals = csv_fold['rescaled disp']
            self.disp_graph_action.setEnabled(True)
            if self.save_loc:
                #sub = "/Profile" if is_file else ""
                filename = file_name.split('.')[0] if is_file else folder_name
                csv_fold.to_excel(f"{self.save_loc}/{filename}.xlsx")
                #self.coords = np.vstack([csv_fold['latitude'].dropna().tolist(), csv_fold['longitude'].dropna().tolist()]).T
                self.coords = csv[['latitude', 'longitude', 'Profile ID', 'rescaled disp', 'actual width']].dropna().reset_index(drop=True)


    def read_h5(self):
        f = h5py.File(self.curr_path, 'r')
        keys = list(f.keys())

        file_name, _ = self.get_path_name(self.curr_path)
        init_p = None
        csv = pd.DataFrame()

        #if dim == 2 then len(keys) needs to be 2. each group then has to have profiles as subgroups
        
        #for i in range(len(keys)):
        for i in range(1):
            ds1 = f[keys[17]]
            ds1 = np.array(ds1[:,np.any(~np.isnan(ds1), axis=0)])

            y1 = np.nanmean(ds1, axis=0)

            #FIX!!!!!!!!
            sigma_x = {}
            x = np.arange(len(y1))

            # x_interp = np.linspace(min(x), max(x), 1000)
            # y_interp = np.interp(x_interp, x, y1)
            # smooth_data = np.array([x_interp, y_interp]).T

            data = np.array([x, y1]).T

            smooth_y1 = gaussian_convolution_nonuniform(x, y1, sigma_x=20)
            smooth_data = np.array([x, smooth_y1]).T

            coords = None
            if self.loc_format:
                coords = self.process_locations(None, i)
                if coords:
                    self.file_names_for_coords.append(file_name) #FIX

            table, fig, model, u, losses, init_p = run_optimization(smooth_data, data, i, self.sigma, self.rand, self.uncert, coords, self.w_bounds)
            self.init_p = init_p if self.prev_ip else None

            self.file_list.setdefault(file_name, {})[i] = (fig, table, model, u, losses, (self.curr_path, i))
            self.show_file_selector()
            csv = pd.concat([csv, table], ignore_index=True)

            if self.save_loc:
                csv.to_excel(self.save_loc + "/" + file_name + ".xlsx")
                savefig(self.fig_dir / f"{file_name}_profile{i}.png")
                close(fig)


    def show_file_selector(self):
        self.clear_layout(self.inner_layout)
        self.clear_layout(self.main_area_layout)
        self.main_area_layout.setAlignment(Qt.AlignTop)
        self.main_area_layout.addWidget(self.scroll_area)
        for file_index in self.file_list:
            btn = QPushButton(file_index)
            btn.setStyleSheet("padding: 10px; border: 1px solid #888; text-align: left;")
            btn.clicked.connect(lambda _, idx=file_index: self.show_profile_selector(idx))
            #self.main_area_layout.addWidget(btn, alignment=Qt.AlignTop)
            self.inner_layout.addWidget(btn, alignment=Qt.AlignTop)

            
    def show_profile_selector(self, file_index, clear=True):
        self.clear_layout(self.processing_panel)

        self.proc_panel_title.setText("PROFILE DISPLAY")
        self.processing_panel.addWidget(self.proc_panel_title)
        self.processing_panel.addWidget(self.title_line)

        back_btn = QPushButton("Back")
        back_btn.clicked.connect(lambda: self.show_file_selector())
        self.processing_panel.addStretch()
        self.processing_panel.addWidget(back_btn)

        if not clear:
            self.clear_layout(self.main_area_layout)
            self.main_area_layout.addWidget(self.scroll_area)
        else:
            self.clear_layout(self.inner_layout)
            self.inner_layout.setAlignment(Qt.AlignTop)

            profiles = self.file_list[file_index]
            for prof_id in profiles.keys():
                btn = QPushButton(f"Profile {prof_id}")
                btn.clicked.connect(lambda _, pidx=prof_id: self.display_profile(file_index, pidx))
                #self.main_area_layout.addWidget(btn, alignment=Qt.AlignTop)
                self.inner_layout.addWidget(btn, alignment=Qt.AlignTop)


    def display_profile(self, file_index, profile_index):

        #self.clear_layout(self.main_area_layout)
        self.main_area_layout.removeWidget(self.scroll_area)
        self.scroll_area.setParent(None)
        self.clear_layout(self.processing_panel)

        table, tabs, hist_button = self.build_profile_view(file_index, profile_index, flag=True)

        self.main_area_layout.addWidget(tabs)

        self.proc_panel_title.setText(f"Parameters for Profile {profile_index}")
        self.processing_panel.addWidget(self.proc_panel_title)
        self.processing_panel.addWidget(self.title_line)

        self.processing_panel.addWidget(table)
        self.processing_panel.addWidget(hist_button)

        # fit_btn = QPushButton("Manual Fit")
        # fit_btn.clicked.connect(lambda: self.create_parameter_sliders(model))

        back_btn = QPushButton("Back")
        back_btn.clicked.connect(lambda: self.show_profile_selector(file_index, clear=False))
        self.processing_panel.addWidget(back_btn)

    def quick_eval(self, file_info):
        file_path, file_ID = file_info
        file = Path(file_path)

        if file.suffix == '.h5':
            pass
        else:
            data = pd.read_csv(file, delimiter=self.delim, header=self.header)
            df = (data[self.cols].rename(columns=self.new_names) if file_ID is None 
                  else data[self.cols].rename(columns=self.new_names).groupby(self.ids).get_group(file_ID))

            return self.eval_data(df, file_ID, history=True)

    
    def show_fit_history(self, tabs, file_info):
        
        model, losses, x_norm, y, scale_shift = self.quick_eval(file_info)

        states = losses["states"]
        model.eval()

        anim_widget = fit_animation.FitHistoryWidget(
            model=model,
            states=states,
            x_norm=x_norm,
            y = y,
            max_points=500
        )

        tabs.addTab(anim_widget, "Fit History Animation")
        tabs.setCurrentWidget(anim_widget)

    def build_profile_view(self, file_index, profile_index, flag=False):
        fig, df, _, u, losses, file_info = self.file_list[file_index][profile_index]

        if flag:
            self.prof_fig = fig

        tabs = QTabWidget()

        canvas_main = FigureCanvas(fig)
        tab1 = QWidget()
        tab1_layout = QVBoxLayout()
        tab1_layout.addWidget(canvas_main)
        tab1.setLayout(tab1_layout)
        tabs.addTab(tab1, "Profile Plot")

        if u is not None:
            canvas_uncert = FigureCanvas(u[0])
            tab2 = QWidget()
            tab2_layout = QVBoxLayout()
            tab2_layout.addWidget(canvas_uncert)
            tab2.setLayout(tab2_layout)
            tabs.addTab(tab2, "Uncertainty")

        fig_loss, ax = subplots(1, 1)
        ax.plot(range(1, len(losses['total_loss']) + 1), losses['total_loss'])
        ax.set_title("Loss vs Epochs")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("MSE Loss (SUM)")
        canvas_loss = FigureCanvas(fig_loss)

        tab3 = QWidget()
        tab3_layout = QVBoxLayout()
        tab3_layout.addWidget(canvas_loss)
        tab3.setLayout(tab3_layout)
        tabs.addTab(tab3, "Loss")

        table = QTableWidget()
        transposed = df.T
        table.setRowCount(transposed.shape[0])
        table.setColumnCount(transposed.shape[1])
        table.setVerticalHeaderLabels(transposed.index.astype(str))

        for i in range(transposed.shape[0]):
            for j in range(transposed.shape[1]):
                item = QTableWidgetItem(str(transposed.iat[i, j]))
                item.setTextAlignment(Qt.AlignCenter)
                table.setItem(i, j, item)

        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        table.setMinimumWidth(300)

        plot_history_button = QPushButton("Show Fit History")
        plot_history_button.clicked.connect(lambda: self.show_fit_history(tabs, file_info))

        return table, tabs, plot_history_button
    
        
    def show_profile_in_popup(self, file_index, profile_index):
        profile_window = QMainWindow()
        profile_window.setWindowTitle(f"Profile {profile_index}")
        profile_window.resize(1000, 800)

        table, tabs = self.build_profile_view(file_index, profile_index, include_table=True)
        profile_widget = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(table, tabs)
        profile_widget.setLayout(layout)
        profile_window.setCentralWidget(profile_widget)

        profile_window.show()
        self.open_windows.append(profile_window)
        profile_window.destroyed.connect(lambda: self.open_windows.remove(profile_window))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QFont("Arial")
    font.setPixelSize(20)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
