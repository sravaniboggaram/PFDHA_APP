import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from os import cpu_count
from csv import Sniffer
from re import findall
from datetime import datetime
import torch
import select_columns
import select_h5_subset
import fit_animation
import create_azimuths
from helper_funs import plot_disp_graph
from maps_functions import generate_google_maps_html, generate_leaflet_html
from FileSelectionModel import SelectorListModel
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QMenuBar, QMenu, QAction, QFileDialog, QSizePolicy, QLineEdit,
    QComboBox, QRadioButton, QButtonGroup, QDialog, QMessageBox, QTableWidget, 
    QTableWidgetItem, QTabWidget, QMainWindow, QSlider, QSplitter,
    QApplication, QCheckBox, QProgressBar, QListView, QStackedWidget,
    QAbstractItemView, QListWidget, QListWidgetItem
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, pyqtSlot, QObject, QUrl, QThread
from profile_evaluationv4 import EvalCoordinator
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtGui import QIcon, QFont
from dataclasses import dataclass, field
from shutil import rmtree
from ResizeImage import ScaledImageLabel


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
    save_loc: list
    file_format: str
    txt_cols_data: list | None
    temp_fig_folder: str
    interp: int | None


@dataclass
class ProcessingRun:
    run_id: int
    name: str
    input_path: Path
    save_loc: str | None

    # Run configuration / output folders
    config: object | None = None
    temp_fig_folder: Path | None = None

    # Main result structures
    file_list: list = field(default_factory=list)
    loss_graph_vals: list = field(default_factory=list)
    disp_graph_vals: list = field(default_factory=list)

    # Profile next/previous navigation cache
    profile_nav_positions: list = field(default_factory=list)
    profile_nav_lookup: dict = field(default_factory=dict)

    # Loss outlier flags
    high_loss_files: set = field(default_factory=set)
    high_loss_profiles: set = field(default_factory=set)

    # Loss tab state
    loss_file_order: list = field(default_factory=list)
    loss_file_checked: dict = field(default_factory=dict)
    loss_x_vals: list = field(default_factory=list)

    # Run state
    completed_profiles: int = 0
    total_profiles: int = 0
    status: str = "created"  # created, running, complete, canceled, error

    # UI state for this run
    selector_view_initialized: bool = False
    selector_mode: str = "files"
    last_file_idx: int | None = None
    last_profile_idx: int | None = None


class CircleButton(QPushButton):
    def __init__(self):
        super().__init__()
        self.setFixedSize(40, 40)
        #self.setStyleSheet("border: none;")
        self.setIcon(QIcon('caltech_logo.png'))
        self.setIconSize(self.size())


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

        self.shutdown_requested = False
        self.force_close = False
        self.shutdown_label = QLabel("Shutting Down ... finishing running profiles")
        self.shutdown_label.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #aa0000; padding: 10px;"
        )
        self.shutdown_label.hide()
        self.cancel_requested = False
        self.processing_active = False
        
        # Column Names
        self.strike = 'strike'
        self.parallel = 'parallel'
        self.perp = 'perpendicular'
        self.ids = 'ids'
        self.az = 'az'
        self.coord_point1 = 'point1' # Latitude or Northing
        self.coord_point2 = 'point2' # Longitude or Easting
        
        self.file_names_for_coords = []
        self.get_loc_fun = None

        self.runs = []
        self.active_run = None
        self.processing_run = None
        self.next_run_id = 1

        self.temp_fig_folder = Path("temp_fig_folder")

        self.curr_path = None
        self.save_loc = None
        self.file_ext = None
        self.loc_file = None
        self.ind = 0

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        
        self.completed_prof_num = 0
        self.num_profs = {}

        self.variable_input = {}

        self.coords = None
        self.open_windows = []
        self.coords_to_fig = {}
        self.model = None
        self.x_data = None
        self.prof_fig = None
        self.API_KEY = "AIzaSyDI7MgDaJ6pHkLVlZIh48y8T-PbL829988"

        self.initUI()


    def closeEvent(self, event):
        # This is the second close triggered by finish_graceful_close().
        # Accept it without asking again.
        if getattr(self, "force_close", False):
            event.accept()
            return

        reply = QMessageBox.question(
            self,
            "Exit Confirmation",
            "Are you sure you want to exit?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            event.ignore()
            return

        # If no evaluation is running, close normally.
        if not hasattr(self, "thread") or self.thread is None or not self.thread.isRunning():
            event.accept()
            return

        # Evaluation is running.
        # Do NOT close yet. Keep the GUI alive so the label can repaint.
        event.ignore()
        self.begin_graceful_shutdown()

    
    def begin_graceful_shutdown(self):
        if getattr(self, "shutdown_requested", False):
            return

        self.shutdown_requested = True

        if self.shutdown_label.parent() is None:
            self.main_area_layout.setAlignment(Qt.AlignCenter)
            self.main_area_layout.addWidget(self.shutdown_label)

        self.shutdown_label.show()

        if self.temp_fig_folder.is_dir():
            rmtree(self.temp_fig_folder)

        try:
            self.start_button.setEnabled(False)
        except Exception:
            pass

        try:
            self.pause_btn.setEnabled(False)
            self.resume_btn.setEnabled(False)
            self.cancel_btn.setEnabled(False)
        except Exception:
            pass

        try:
            if hasattr(self, "worker") and self.worker is not None:
                self.worker.shutdown_after_running_jobs()
        except Exception as e:
            print("Shutdown request failed:", e)


    def finish_graceful_close(self):
        for window in list(getattr(self, "open_windows", [])):
            try:
                window.close()
            except Exception:
                pass

        try:
            if hasattr(self, "thread") and self.thread is not None:
                self.thread.quit()
                self.thread.wait(3000)
        except Exception:
            pass

        self.force_close = True
        self.close()

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

        loss_graph = QAction("View Losses", self)
        loss_graph.triggered.connect(self.show_losses_tab)
        tools_menu.addAction(loss_graph)

        self.disp_graph_action = QAction("Displacement Graph", self)
        self.disp_graph_action.setEnabled(False)
        self.disp_graph_action.triggered.connect(self.display_disp_graph)
        tools_menu.addAction(self.disp_graph_action)

        menu_bar.addMenu(file_menu)
        menu_bar.addMenu(tools_menu)
        menu_bar.addMenu(help_menu)

        self.run_selector = QComboBox()
        self.run_selector.currentIndexChanged.connect(self.on_run_selector_changed)

        circle_btn = CircleButton()

        # Top Layout
        top_widget = QWidget()
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)
        top_layout.addWidget(circle_btn)
        top_layout.addWidget(menu_bar)
        top_layout.addWidget(self.run_selector)
        top_layout.addStretch()
        top_widget.setLayout(top_layout)
        top_widget.setFixedHeight(menu_bar.sizeHint().height() + 10)

        self.create_proc_panel_widgets()

        self.processing_panel = QVBoxLayout()

        self.profile_screen_split = QSplitter(Qt.Horizontal)
        
        self.left_frame = QFrame()
        self.left_frame.setFrameShape(QFrame.Box)
        self.left_frame.setLayout(self.processing_panel)
        #self.left_frame.setFixedWidth(450)
        self.left_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.profile_screen_split.addWidget(self.left_frame)
        
        # Main Area
        self.main_area = QFrame()
        self.main_area.setFrameShape(QFrame.Box)
        self.main_area.setStyleSheet("background-color: #f0f0f0;")
        self.main_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_area_layout = QVBoxLayout()
        self.main_area.setLayout(self.main_area_layout)
        self.profile_screen_split.addWidget(self.main_area)

        # Left Panel
        self.open_proc()

        # Final Layout
        self.main_content_layout = QVBoxLayout()
        self.main_content_layout.addWidget(top_widget)
        self.main_content_layout.addWidget(self.profile_screen_split)
        self.setLayout(self.main_content_layout)
        self.profile_screen_split.setSizes([300,900])


    def add_run_to_selector(self, run):
        self.run_selector.blockSignals(True)
        self.run_selector.addItem(run.name, run.run_id)
        self.run_selector.setCurrentIndex(self.run_selector.count() - 1)
        self.run_selector.blockSignals(False)

    def on_run_selector_changed(self, index):
        if index < 0:
            return

        run_id = self.run_selector.itemData(index)

        for run in self.runs:
            if run.run_id == run_id:
                self.set_active_run(run)
                return

    def show_losses_tab(self):
        """
        Replace the central profile workspace with a tabbed workspace:
            Tab 1: existing profile/file/plot UI
            Tab 2: loss plotting UI
        """

        # Already created, just switch to Losses tab
        if hasattr(self, "results_tabs") and self.results_tabs is not None:
            self.results_tabs.setCurrentWidget(self.losses_tab)
            return

        # Remove existing profile workspace from its current parent layout.
        # Replace `self.main_content_layout` with the layout that currently owns
        # the central display area.
        self.main_content_layout.removeWidget(self.profile_screen_split)

        self.results_tabs = QTabWidget()

        self.results_tabs.addTab(self.profile_screen_split, "Profiles")

        self.losses_tab = QWidget()
        self.losses_tab_layout = QVBoxLayout(self.losses_tab)
        self.losses_tab_layout.setContentsMargins(8, 8, 8, 8)

        self.build_losses_tab()

        self.results_tabs.addTab(self.losses_tab, "Losses")

        self.main_content_layout.addWidget(self.results_tabs)

        self.results_tabs.setCurrentWidget(self.losses_tab)

    def get_list_mode(self, list_type):
        """
        Returns:
            "folder_single_profiles"
            "single_file_multi_profiles"
            "folder_multi_profiles"
            "empty"
        """
        if list_type == "loss":
            if self.active_run is None or not self.active_run.loss_graph_vals:
                return "empty"
            vals_list = self.active_run.loss_graph_vals

        has_lists = any(isinstance(item, list) for item in vals_list)
        has_single_items = any(not isinstance(item, list) for item in vals_list)

        if has_lists and len(vals_list) == 1:
            return "single_file_multi_profiles"

        if has_lists:
            return "folder_multi_profiles"

        if has_single_items:
            return "folder_single_profiles"

        return "empty"
    

    def build_losses_tab(self):
        self.clear_layout_delete(self.losses_tab_layout)

        mode = self.get_list_mode("loss")

        if mode == "empty":
            label = QLabel("No losses are available yet.")
            label.setAlignment(Qt.AlignCenter)
            self.losses_tab_layout.addWidget(label)
            return

        if mode == "folder_single_profiles" or mode == "single_file_multi_profiles":

            if mode == "single_file_multi_profiles":
                vals_list = self.active_run.loss_graph_vals[0]
                loss_x_vals = self.active_run.loss_x_vals[0]
            else:
                vals_list = self.active_run.loss_graph_vals
                loss_x_vals = self.active_run.loss_x_vals

            fig = self.make_folder_or_single_nprof_fig(loss_x_vals, vals_list)
            canvas = FigureCanvas(fig)
            self.losses_tab_layout.addWidget(canvas)
            return

        if mode == "folder_multi_profiles":
            self.build_multi_file_loss_selector()
            return
        

    def make_folder_or_single_nprof_fig(self, loss_x_vals, vals_list):

        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)

        ax.tick_params(axis='x', labelrotation=90)
        ax.plot(loss_x_vals, vals_list, marker="o")
        ax.set_title("Final Loss by File")
        ax.set_xlabel("File")
        ax.set_ylabel("Final Loss")
        ax.grid(True)

        fig.tight_layout()
        return fig
    

    def init_loss_file_selection_state(self):
        run = self.active_run
        if run is None:
            return

        n_files = len(run.loss_graph_vals)

        # Initialize if this run has never initialized loss ordering,
        # or if the structure changed.
        if len(run.loss_file_order) != n_files:
            run.loss_file_order = list(range(n_files))

        # Add missing checkbox states, but preserve existing choices.
        for file_idx in run.loss_file_order:
            if file_idx not in run.loss_file_checked:
                run.loss_file_checked[file_idx] = True

        # Remove stale checkbox states if needed.
        run.loss_file_checked = {
            file_idx: checked
            for file_idx, checked in run.loss_file_checked.items()
            if file_idx in run.loss_file_order
        }

    def build_multi_file_loss_selector(self):
        self.init_loss_file_selection_state()

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self.loss_file_list_widget = QListWidget()
        self.loss_file_list_widget.setSelectionMode(QAbstractItemView.SingleSelection)

        self.populate_loss_file_list_widget()

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.addWidget(QLabel("Select files and order"))
        left_layout.addWidget(self.loss_file_list_widget)

        self.loss_plot_container = QWidget()
        self.loss_plot_layout = QVBoxLayout(self.loss_plot_container)
        self.loss_plot_layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(left_panel, stretch=1)
        layout.addWidget(self.loss_plot_container, stretch=3)

        self.losses_tab_layout.addWidget(container)

        self.update_multi_file_loss_plot_from_list()


    def get_file_display_name_for_loss(self, file_idx):
        file_item = self.active_run.file_list[file_idx]

        if isinstance(file_item, list):
            first_done = next((p for p in file_item if p is not None), None)
            if first_done is not None:
                return first_done["file_key"][0]

        elif file_item is not None:
            return file_item["file_key"][0]

        return f"File {file_idx + 1}"
    
    def set_loss_file_checked(self, file_idx, state):
        self.active_run.loss_file_checked[file_idx] = state == Qt.Checked
        self.update_multi_file_loss_plot_from_list()
    
    def populate_loss_file_list_widget(self):
        self.loss_file_list_widget.clear()

        for row, file_idx in enumerate(self.active_run.loss_file_order):
            item = QListWidgetItem()
            item.setData(Qt.UserRole, file_idx)

            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(4, 2, 4, 2)

            checkbox = QCheckBox(self.get_file_display_name_for_loss(file_idx))
            checkbox.setChecked(self.active_run.loss_file_checked.get(file_idx, True))

            up_btn = QPushButton("↑")
            down_btn = QPushButton("↓")

            up_btn.setFixedWidth(32)
            down_btn.setFixedWidth(32)

            up_btn.setEnabled(row > 0)
            down_btn.setEnabled(row < len(self.active_run.loss_file_order) - 1)

            checkbox.stateChanged.connect(
                lambda state, idx=file_idx: self.set_loss_file_checked(idx, state)
            )

            up_btn.clicked.connect(
                lambda checked=False, idx=file_idx: self.move_loss_file(idx, -1)
            )

            down_btn.clicked.connect(
                lambda checked=False, idx=file_idx: self.move_loss_file(idx, 1)
            )

            row_layout.addWidget(checkbox)
            row_layout.addStretch()
            row_layout.addWidget(up_btn)
            row_layout.addWidget(down_btn)

            item.setSizeHint(row_widget.sizeHint())

            self.loss_file_list_widget.addItem(item)
            self.loss_file_list_widget.setItemWidget(item, row_widget)

    def move_loss_file(self, file_idx, direction):
        """
        direction = -1 moves up
        direction =  1 moves down
        """

        if file_idx not in self.active_run.loss_file_order:
            return

        old_pos = self.active_run.loss_file_order.index(file_idx)
        new_pos = old_pos + direction

        if new_pos < 0 or new_pos >= len(self.active_run.loss_file_order):
            return

        self.active_run.loss_file_order[old_pos], self.active_run.loss_file_order[new_pos] = (
            self.active_run.loss_file_order[new_pos],
            self.active_run.loss_file_order[old_pos],
        )

        self.populate_loss_file_list_widget()
        self.update_multi_file_loss_plot_from_list()


    def update_multi_file_loss_plot_from_list(self):
        run = self.active_run
        if run is None:
            return

        if not hasattr(self, "loss_plot_layout"):
            return

        selected_file_indices = [
            file_idx
            for file_idx in run.loss_file_order
            if run.loss_file_checked.get(file_idx, True)
        ]

        fig = self.make_multi_file_loss_fig(selected_file_indices, run=run)

        self.clear_layout_delete(self.loss_plot_layout)

        canvas = FigureCanvas(fig)
        self.loss_plot_layout.addWidget(canvas)
    

    def make_multi_file_loss_fig(self, selected_file_indices, run=None):
        if run is None:
            run = self.active_run

        fig = Figure(figsize=(9, 5))
        ax = fig.add_subplot(111)

        if run is None:
            ax.text(
                0.5, 0.5,
                "No active run",
                ha="center",
                va="center",
                transform=ax.transAxes
            )
            return fig

        x_vals = []
        y_vals = []

        for file_idx in selected_file_indices:
            y_vals.extend(run.loss_graph_vals[file_idx])
            x_vals.extend(run.loss_x_vals[file_idx])

            # for loss_idx in range(len(file_losses)):
            #     loss = file_losses[loss_idx]
            #     x_vals.append("Profile " + run.file_list[file_idx][loss_idx]['file_num'])
            #     y_vals.append(loss)

        if x_vals:
            ax.plot(x_vals, y_vals, marker="o")
        else:
            ax.text(
                0.5, 0.5,
                "No completed profile losses yet",
                ha="center",
                va="center",
                transform=ax.transAxes
            )

        ax.tick_params(axis='x', labelrotation=90)
        ax.set_title("Final Loss by Selected Files", fontdict={'size': 40})
        ax.set_xlabel("Profiles in selected file order", fontdict={'size': 28})
        ax.set_ylabel("Final Loss", fontdict={'size': 28})
        ax.grid(True)

        fig.tight_layout()
        return fig


    def open_map_view(self):
        from PyQt5.QtWebEngineWidgets import QWebEngineView
        
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

        # CPU
        cpu_layout = QHBoxLayout()
        radio_cpu = QRadioButton()
        label_cpu = QLabel("CPU")
        label_cpu.setStyleSheet("margin-left: 5px;")
        info_cpu = QLabel("ⓘ")
        info_cpu.setToolTip("Computation happens on CPU")
        info_cpu.setStyleSheet("color: black; margin-left: 5px;")

        # GPU
        gpu_layout = QHBoxLayout()
        radio_gpu = QRadioButton()
        label_gpu = QLabel("GPU")
        label_gpu.setStyleSheet("margin-left: 5px;")
        info_gpu = QLabel("ⓘ")
        info_gpu.setToolTip("Computation happens on CPU")
        info_gpu.setStyleSheet("color: black; margin-left: 5px;")

        # CPU/GPU Button Group
        cpu_gpu_group = QButtonGroup(self)
        cpu_gpu_group.addButton(radio_cpu, id=0)
        cpu_gpu_group.addButton(radio_gpu, id=1)
        radio_cpu.setChecked(True)

        self.variable_input['GPU'] = cpu_gpu_group

        # Number of cores
        cores_label = QLabel("Number of CPU/GPU cores")
        cores_label.setWordWrap(True)
        cores_line_edit = QLineEdit()
        cores_line_edit.setPlaceholderText("8")
        self.variable_input['n_cores'] = cores_line_edit

        # Peak smoothing sigma
        peak_sigma_label = QLabel("Set \u03C3 for smoothing")
        peak_sigma_label.setWordWrap(True)
        peak_sigma_line_edit = QLineEdit()
        peak_sigma_line_edit.setPlaceholderText("Enter Value")
        self.variable_input['sigma'] = peak_sigma_line_edit
        
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

        params_widgets = [params_title, {cpu_layout: [radio_cpu, label_cpu, info_cpu]}, 
                          {gpu_layout: [radio_gpu, label_gpu, info_gpu]}, cores_label,
                          cores_line_edit, peak_sigma_label, peak_sigma_line_edit, 
                          dim_label, combo_dim, interp_label, interp_n]


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
        self.clear_layout_detach(self.proc_panel_stack_layout)
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

    def _hline(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("color: #aaa;")
        return line

    def open_proc(self):
        if self.proc_panel_title.text() != "PROCESSING PANEL":
            self.clear_layout_detach(self.processing_panel)
            self.ind = 0

            self.proc_panel_title.setText("PROCESSING PANEL")
            self.processing_panel.addWidget(self.proc_panel_title)
            self.processing_panel.addWidget(self.title_line)
            self.processing_panel.addWidget(self.proc_panel_stack)

            self.display_proc_widgets()
            self.processing_panel.addStretch()
            
            self.next_back_layout.addWidget(self.proc_panel_next, alignment=Qt.AlignRight)
            self.processing_panel.addLayout(self.next_back_layout)
        

    def check_inputs_entered(self, button):
        all_valid = True
        if self.curr_path == None:
            all_valid = False

        button.setEnabled(all_valid)


    def setup_profile_display_area(self):
        self.main_stack = QStackedWidget()

        # Page 1: selector
        self.selector_page = QWidget()
        self.selector_page_layout = QVBoxLayout()
        self.selector_page_layout.setContentsMargins(0, 0, 0, 0)
        self.selector_page.setLayout(self.selector_page_layout)

        if not hasattr(self, "selector_view"):
            self.setup_selector_view()

        self.selector_page_layout.addWidget(self.selector_view)

        # Page 2: profile display
        self.profile_page = QWidget()
        self.profile_page_layout = QVBoxLayout()
        self.profile_page_layout.setContentsMargins(0, 0, 0, 0)
        self.profile_page.setLayout(self.profile_page_layout)

        self.main_stack.addWidget(self.selector_page)
        self.main_stack.addWidget(self.profile_page)

        self.main_area_layout.addWidget(self.main_stack)


    def clear_layout_detach(self, layout):
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()

            if widget:
                layout.removeWidget(widget)
                widget.setParent(None)
            elif child_layout:
                self.clear_layout_detach(child_layout)
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
                self.clear_layout_detach(child_layout)
                self.show_in_layout(child_layout, item[child_layout])
                layout.addLayout(child_layout)
        layout.addStretch()


    def clear_layout_delete(self, layout, keep_widgets=None):
        if layout is None:
            return

        if keep_widgets is None:
            keep_widgets = set()

        while layout.count():
            item = layout.takeAt(0)

            widget = item.widget()
            child_layout = item.layout()

            if widget is not None:
                if widget in keep_widgets:
                    widget.setParent(None)
                else:
                    widget.setParent(None)
                    widget.deleteLater()

            elif child_layout is not None:
                self.clear_layout_delete(child_layout, keep_widgets=keep_widgets)
                child_layout.setParent(None)


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
                #timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                #self.fig_dir = Path(self.save_loc) / f"figures_{timestamp}"
                #self.fig_dir.mkdir(parents=True, exist_ok=True)
            else:
                self.import_label.setText(folder_path)
                self.curr_path = Path(folder_path)

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

    def make_run_temp_folder(self, run_id):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = self.temp_fig_folder / f"run_{run_id}_{timestamp}"
        folder.mkdir(parents=True, exist_ok=True)
        return folder


    def create_new_run(self):
        run_id = self.next_run_id
        self.next_run_id += 1

        name = f"Run {run_id}"

        run = ProcessingRun(
            run_id=run_id,
            name=name,
            input_path=self.curr_path,
            save_loc=self.save_loc,
        )

        run.temp_fig_folder = self.make_run_temp_folder(run_id)

        self.active_run = run

        return run


    def set_active_run(self, run):
        self.active_run = run

        # Refresh visible selector/losses for this run.
        if hasattr(self, "selector_model"):
            self.refresh_file_selector_after_result()

        if hasattr(self, "losses_tab") and self.losses_tab is not None:
            self.build_losses_tab()


    def process(self):
        if self.processing_run is not None and self.processing_run.status == "running":
            QMessageBox.information(
                self,
                "Processing Active",
                "A run is already processing. Please wait for it to finish or cancel it before starting another run.",
                QMessageBox.Ok
            )
            return

        self.n_dim = int(self.variable_input['n_dim'].currentText())
        #fix
        self.sigma = float(self.variable_input['sigma'].text()) if len(self.variable_input['sigma'].text())>0 else 10
        self.set_loc_format()
        self.rand = bool(self.variable_input['IP'].checkedId())
        self.uncert = bool(self.variable_input['Mode'].checkedId())
        interp = int(self.variable_input['interp'].text()) if len(self.variable_input['interp'].text())>0 else None
        w_min = float(self.variable_input['w_min'].text()) if len(self.variable_input['w_min'].text())>0 else None
        w_max = float(self.variable_input['w_max'].text()) if len(self.variable_input['w_max'].text())>0 else None
        self.w_bounds = None if w_min is None and w_max is None else [w_min, w_max]
        gpu = bool(self.variable_input['GPU'].checkedId())
        n_cores = int(self.variable_input['n_cores'].text()) if len(self.variable_input['n_cores'].text())>0 else 8
        self.init_p = None

        if gpu:
            if torch.cuda.is_available():
                device = "cuda"
                if n_cores > torch.cuda.device_count():
                    QMessageBox.information(
                        self,
                        f"Not Enough GPUs",
                        f"Only {torch.cuda.device_count()} GPUs available. Go back and choose a lower number",
                        QMessageBox.Ok
                    )
                    return
            else:
                QMessageBox.information(
                    self,
                    "GPU Not Available",
                    "GPU is not available on this device. Go back and select CPU",
                    QMessageBox.Ok
                )
                return
        else:
            device = "cpu"
            if n_cores > cpu_count():
                reply = QMessageBox.information(
                    self,
                    f"Not Enough CPUs",
                    f"Only {cpu_count()} CPUs available. Go back and choose a lower number",
                    QMessageBox.Ok
                )
                return

        is_file = self.curr_path.is_file()
        folder = [self.curr_path] if is_file else sorted(list(self.curr_path.glob("*")), key=str)
        num_files = len(folder)

        (folder_name, _) = self.get_path_name(self.curr_path) if not is_file else (None, None)

        suffix = folder[0].suffix

        run = self.create_new_run()
        self.processing_run = run
        run.status = "running"
        
        if suffix == '.h5':
            jobs = self.read_h5(folder, num_files, run)
            cols_data = None
        else:
            jobs, cols_data = self.read_txt(is_file, folder, num_files, folder_name, run)
        
        if jobs is None:
            self.processing_run = None
            self.next_run_id -= 1
            return
        
        self.active_run = run
        self.add_run_to_selector(run)
        self.runs.append(run)

        self.build_profile_nav_cache()
        
        self.config = EvalConfig(
            strike=self.strike,
            parallel=self.parallel,
            perp=self.perp,
            ids=self.ids,
            az=self.az,
            coord_point1=self.coord_point1,
            coord_point2=self.coord_point2,
            n_dim=self.n_dim,
            sigma=self.sigma,
            rand=self.rand,
            uncert=self.uncert,
            w_bounds=self.w_bounds,
            loc_format=self.loc_format,
            save_loc=self.save_loc,
            file_format=suffix,
            txt_cols_data=cols_data,
            temp_fig_folder=run.temp_fig_folder,
            interp=interp
        )

        run.config = self.config

        self.start_evaluation(jobs, device, n_cores)

    def build_profile_nav_cache(self):
        """
        Creates a flat navigation list with the same logical order as self.file_list.

        Each entry is initially None.
        When a profile finishes, the matching entry becomes:
            (file_idx, None)         for single-profile files
            (file_idx, profile_idx)  for multi-profile files
        """

        self.active_run.profile_nav_positions = []
        self.active_run.profile_nav_lookup = {}

        for file_idx, file_item in enumerate(self.active_run.file_list):
            if isinstance(file_item, list):
                for profile_idx in range(len(file_item)):
                    flat_idx = len(self.active_run.profile_nav_positions)

                    self.active_run.profile_nav_positions.append(None)
                    self.active_run.profile_nav_lookup[(file_idx, profile_idx)] = flat_idx

            else:
                flat_idx = len(self.active_run.profile_nav_positions)

                self.active_run.profile_nav_positions.append(None)
                self.active_run.profile_nav_lookup[(file_idx, None)] = flat_idx

    def get_h5_subset(self, file_names):
        dialog = select_h5_subset.SelectH5Subset(file_names)
        if dialog.exec_() == QDialog.Accepted:
            subset_text = dialog.selected_options()
            if subset_text is None: # All radio button was chosen
                return "ALL"
            for i in range(len(subset_text)): # Each file
                subset_text[i] = subset_text[i].split(",")
                for j in range(len(subset_text[i])):
                    subset_text[i][j] = subset_text[i][j].strip()
            return subset_text
        return None


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

        while dialog.exec_() == QDialog.Accepted:
        #if dialog.exec_() == QDialog.Accepted:
            option_strike, option_parallel, option_perpendicular, prof_id, point1, point2, az = dialog.selected_options()

            options_text = f"<b>Distance Along Strike:</b> {option_strike}\n<b>Parallel Displacement:</b> {option_parallel}"
            if option_perpendicular:
                options_text += f"\n<b>Perpendicular Displacement:</b> {option_perpendicular}"
            (text1, text2) = ("<b>Latitude:</b>", "<b>Longitude:</b>") if self.loc_format == 'LATLON' else ("<b>Northing:</b>", "<b>Easting:</b>")
            options_text += f"\n<b>Profile ID:</b> {prof_id} \n {text1} {point1} \n {text2} {point2} \n<b>Azimuth:</b> {az}"

            # msg = QMessageBox()
            # msg.setTextFormat(Qt.RichText)
            # msg.information(self, "Columns Selected", options_text, QMessageBox.Ok)
            # back_btn = msg.addButton("Back", QMessageBox.ActionRole)
            # if msg.clickedButton() == back_btn:
            #     dialog = select_columns.SelectColumns(data.columns, self.n_dim, self.variable_input['loc_inp'].checkedId() == 1, self.loc_format)
            #     continue

            reply = QMessageBox()
            reply.setWindowTitle("Columns Selected")
            reply.setText(options_text)
            back_btn = reply.addButton("Back", QMessageBox.ActionRole)
            ok_btn = reply.addButton("Ok", QMessageBox.ActionRole)

            reply.exec_()

            if reply.clickedButton() == back_btn:
                dialog = select_columns.SelectColumns(data.columns, self.n_dim, self.variable_input['loc_inp'].checkedId() == 1, self.loc_format)
                continue

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
            
            return [cols, new_names, prof_id, delim, header]
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
    

    def read_txt(self, is_file, folder, num_files, folder_name, run):
        test_file = self.curr_path if is_file else next(self.curr_path.iterdir())

        cols_data = self.get_txt_cols(test_file)
        if cols_data is None:
            return None, None

        cols, new_names, prof_id, delim, header = cols_data[0], cols_data[1], cols_data[2], cols_data[3], cols_data[4]
        self.num_profs[folder_name] = 0
        single_prof_count = 0
        n_profs_counts = []

        jobs = []
        i = 0
        while i < num_files:
            file = folder[i]
            file_name, file_num = self.get_path_name(file)
            data = pd.read_csv(file, delimiter=delim, header=header)
            df = data[cols].rename(columns=new_names)

            # Single Profile in File
            if prof_id == "None":
                run.loss_x_vals.append(file_name)
                coords = self.process_locations(df.iloc[0], i) if self.loc_format else None
                jobs.append({'df': str(file),
                             'file_num': file_name, 
                             'coords': coords, 
                             'file_key': (file_name, single_prof_count, None),
                             'file_info': (file, None)})
                single_prof_count += 1
                
            else:
                profiles = df.groupby(self.ids)
                profile_ids = list(profiles.groups.keys())[:10]
                num_ids = len(profile_ids)
                n_profs_counts.append(num_ids)
                run.loss_x_vals.append([])

                for id_idx in range(num_ids):
                    run.loss_x_vals[-1].append("Profile " + curr_prof_id)
                    curr_prof_id = profile_ids[id_idx]
                    orig_prof = profiles.get_group(curr_prof_id)
                    coords = self.process_locations(orig_prof.iloc[0], curr_prof_id) if self.loc_format else None
                    jobs.append({'df': str(file), 
                                 'file_num': curr_prof_id, 
                                 'coords': coords, 
                                 'file_key': (file_name, id_idx, i), #file_name, profile index, file index
                                 'file_info': (file, curr_prof_id)})
 

            i += 1

        run.file_list = single_prof_count*[None]
        run.disp_graph_vals = single_prof_count*[None]
        run.loss_graph_vals = single_prof_count*[None]
        t = 0
        for count in n_profs_counts:
            t += count
            run.file_list.append(count*[None])
            run.disp_graph_vals.append(count*[None])
            run.loss_graph_vals.append(count*[None])

        run.total_profiles = num_files if prof_id == "None" else t
        #self.progress_bar.setMaximum(total_profs_num)

        return jobs, cols_data


    def read_h5(self, folder, n_files, run):
        file_names = [self.get_path_name(f)[0] for f in folder]
        subset = self.get_h5_subset(file_names)
        if subset is None:
            return None
        
        jobs = []
        for curr_file_i in range(n_files):
            f_path = folder[curr_file_i]
            f = h5py.File(f_path, 'r')
            file_keys = list(f.keys())

            groups = [f[k] for k in file_keys]
            prof_keys = list(groups[0].keys())
            run.disp_graph_vals = [[] for _ in range(len(groups))]
            run.loss_x_vals.append([])

            if isinstance(subset, list) and subset[curr_file_i] != "":
                keys = []
                for text in subset[curr_file_i]:
                    profs = text.split("-")
                    profs = [prof.strip() for prof in profs]
                    if len(profs) == 1:
                        try:
                            keys.append(prof_keys.index(profs[0]))
                        except:
                            QMessageBox.information(
                                self,
                                "Invalid Keys",
                                profs[0] + " not in " + file_names[curr_file_i] +  " keys",
                                QMessageBox.Ok
                            )
                            return None
                    else:
                        try:
                            keys.extend(list(range(prof_keys.index(profs[0]),
                                                   prof_keys.index(profs[1]))))
                        except:
                            QMessageBox.information(
                                self,
                                "Invalid Keys",
                                profs[0] + " or " + profs[1] + " not in " + file_names[curr_file_i] +  " keys",
                                QMessageBox.Ok
                            )
                            return None
                keys.sort()
                        
            else:
                keys = range(len(prof_keys))
            
            run.total_profiles = len(keys)
            #self.progress_bar.setMaximum(len(keys))

            file_name, _ = self.get_path_name(f_path)

            run.file_list.append(len(keys)*[None])
            run.disp_graph_vals.append(len(keys)*[None])
            run.loss_graph_vals.append(len(keys)*[None])
            profile_list_idx = 0
            for k in keys:
                key = prof_keys[k]
                print("KEY ", key, key[key.rfind("_") + 1:])

                run.loss_x_vals[-1].append(key)
                
                coords = self.process_locations(None, k) if self.loc_format else None
                jobs.append({'df_path': f_path,
                             'prof_key': key,
                             'file_num': key[key.rfind("_") + 1:], # profile_id format: profile_num
                             'coords': coords, 
                             'file_key': (file_name, profile_list_idx, curr_file_i),
                             'file_info': (f_path, k)})
                profile_list_idx += 1
        return jobs
    
    def pause_evaluation(self):
        if hasattr(self, "worker") and self.worker is not None:
            self.worker.pause()

        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)


    def resume_evaluation(self):
        if hasattr(self, "worker") and self.worker is not None:
            self.worker.resume()

        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)


    def cancel_evaluation(self):
        reply = QMessageBox.question(
            self,
            "Cancel Processing",
            "Cancel profile processing?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        self.cancel_requested = True

        if self.shutdown_label.parent() is None:
            self.main_area_layout.insertWidget(0, self.shutdown_label)

        self.shutdown_label.setText("Canceling... stopping queued profile calculations.")
        self.shutdown_label.show()

        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)

        if hasattr(self, "worker") and self.worker is not None:
            self.worker.cancel()

    def on_eval_paused(self):
        self.shutdown_label.setText(
            "Paused. Running profiles may finish, but no new profiles will start."
        )
        if self.shutdown_label.parent() is None:
            self.main_area_layout.insertWidget(0, self.shutdown_label)
        self.shutdown_label.show()


    def on_eval_resumed(self):
        if not getattr(self, "shutdown_requested", False):
            self.shutdown_label.hide()
    
    def ensure_eval_controls(self):
        if hasattr(self, "eval_controls_widget"):
            if self.eval_controls_widget.parent() is None:
                self.main_area_layout.insertWidget(0, self.eval_controls_widget)
            self.eval_controls_widget.show()
            return

        self.eval_controls_widget = QWidget()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignHCenter)
        self.eval_controls_widget.setLayout(layout)

        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.cancel_btn = QPushButton("Cancel")

        self.pause_btn.clicked.connect(self.pause_evaluation)
        self.resume_btn.clicked.connect(self.resume_evaluation)
        self.cancel_btn.clicked.connect(self.cancel_evaluation)

        layout.addWidget(self.pause_btn)
        layout.addWidget(self.resume_btn)
        layout.addWidget(self.cancel_btn)
        layout.addStretch()

        self.resume_btn.setEnabled(False)

        self.main_area_layout.setAlignment(Qt.AlignTop)
        self.main_area_layout.insertWidget(0, self.eval_controls_widget)
        self.main_area_layout.addWidget(self.progress_bar)
        self.progress_bar.show()
        self.eval_controls_widget.show()


    def create_loss_flags(self, run):
        flat = []
        losses = []
        for file_idx, item in enumerate(run.loss_graph_vals):

            # Multi-profile file
            if isinstance(item, list):
                for profile_idx, loss in enumerate(item):
                    if loss is None:
                        continue

                    flat.append({
                        "file_idx": file_idx,
                        "profile_idx": profile_idx,
                        "loss": float(loss),
                    })
                    losses.append(float(loss))
                    if profile_idx == 10:
                        losses[-1] = 0.6

            # Single-profile file
            else:
                if item is None:
                    continue

                flat.append({
                    "file_idx": file_idx,
                    "profile_idx": None,
                    "loss": float(item),
                })
                losses.append(float(item))

        if len(flat) <= 20:
            return set(), set()
        
        window_radius=5
        ratio_threshold=3.0
        min_abs_delta=0.0

        high_loss_profs = set()
        high_loss_files = set()

        for i, row in enumerate(flat):
            left = max(0, i - window_radius)
            right = min(len(losses), i + window_radius + 1)

            neighbor_losses = np.concatenate([
                losses[left:i],
                losses[i + 1:right],
            ])

            if neighbor_losses.size < 3:
                continue

            local_median = np.median(neighbor_losses)

            if local_median <= 0:
                continue
            
            # print("LOSS I ,     RATIO,      DELTA")
            # print("----------------------------------")
            # print(losses[i], "      ", (ratio_threshold * local_median), "      ", (losses[i] - local_median))
            ratio_bad = losses[i] > (ratio_threshold * local_median)
            delta_bad = (losses[i] - local_median) > min_abs_delta

            if ratio_bad and delta_bad:
                high_loss_profs.add((row["file_idx"], row["profile_idx"]))
                high_loss_files.add(row["file_idx"])

        #The model needs access to the latest flag sets.
        self.selector_model.high_loss_files = high_loss_files
        self.selector_model.high_loss_profiles = high_loss_profs

        top_left = self.selector_model.index(0, 0)
        bottom_right = self.selector_model.index(
            max(0, self.selector_model.rowCount() - 1),
            0
        )

        self.selector_model.dataChanged.emit(top_left, bottom_right)

        return high_loss_files, high_loss_profs

    def start_evaluation(self, jobs, device, n_cores):
        print(datetime.now())

        self.shutdown_requested = False
        self.cancel_requested = False
        self.force_close = False
        self.processing_active = True

        self.progress_bar.setMaximum(self.processing_run.total_profiles)
        self.progress_bar.setValue(0)
        self.progress_bar.show()

        self.ensure_eval_controls()
        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        self.thread = QThread()
        self.worker = EvalCoordinator(jobs, self.config, device, n_cores)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)

        run = self.processing_run

        self.worker.result_ready.connect(
            lambda result, run=run: self.on_profile_result_ready_for_run(run, result)
        )

        self.worker.progress.connect(
            lambda completed, run=run: self.on_profile_progress_for_run(run, completed)
        )

        self.worker.finished.connect(
            lambda results, run=run: self.on_eval_worker_finished_for_run(run, results)
        )

        self.worker.error.connect(self.on_eval_error)

        self.worker.finished.connect(self.thread.quit)

        self.worker.paused.connect(self.on_eval_paused)
        self.worker.resumed.connect(self.on_eval_resumed)

        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.on_eval_thread_finished)

        self.thread.start()

    def on_eval_thread_finished(self):
        self.worker = None
        self.thread = None

        # self.pause_btn.setEnabled(False)
        # self.resume_btn.setEnabled(False)
        # self.cancel_btn.setEnabled(False)

        self.pause_btn.setParent(None)
        self.resume_btn.setParent(None)
        self.cancel_btn.setParent(None)


    def on_profile_progress_for_run(self, run, completed):
        run.completed_profiles = completed

        # Only update the visible progress bar if this is the displayed run.
        if self.active_run is run:
            self.progress_bar.setValue(completed)


    def on_profile_result_ready_for_run(self, run, r):
        _, prof_idx, file_idx = r["file_key"]

        if file_idx is None:
            run.file_list[prof_idx] = r
            run.loss_graph_vals[prof_idx] = r["final_loss"]
            nav_key = (prof_idx, None)
        else:
            run.file_list[file_idx][prof_idx] = r
            run.loss_graph_vals[file_idx][prof_idx] = r["final_loss"]
            nav_key = (file_idx, prof_idx)

        flat_idx = run.profile_nav_lookup.get(nav_key)
        if flat_idx is not None:
            run.profile_nav_positions[flat_idx] = nav_key

        # Only refresh visible GUI if the run being updated is currently displayed.
        if self.active_run is run:
            if hasattr(self, "losses_tab") and self.losses_tab is not None:
                self.refresh_losses_tab_after_result()

            if not run.selector_view_initialized:
                self.show_file_selector_first_run()
                run.selector_view_initialized = True
            else:
                self.refresh_file_selector_after_result(r)


    def on_eval_worker_finished_for_run(self, run, results):
        self.processing_active = False

        if self.processing_run is run:
            self.processing_run = None

        if self.shutdown_requested:
            self.finish_graceful_close()
            return

        if self.cancel_requested:
            run.status = "canceled"

            if self.active_run is run:
                self.shutdown_label.setText(
                    "Processing canceled. Running profiles may finish, but no new profiles will start."
                )
                self.pause_btn.setEnabled(False)
                self.resume_btn.setEnabled(False)
                self.cancel_btn.setEnabled(False)

            return

        run.status = "complete"
        run.high_loss_files, run.high_loss_profiles = self.create_loss_flags(run)

        if self.active_run is run:
            try:
                self.pause_btn.setEnabled(False)
                self.resume_btn.setEnabled(False)
                self.cancel_btn.setEnabled(False)
            except Exception:
                pass

            self.shutdown_label.hide()
            self.refresh_file_selector_after_result()

            if hasattr(self, "losses_tab") and self.losses_tab is not None:
                self.build_losses_tab()


    def refresh_losses_tab_after_result(self):
        if not hasattr(self, "results_tabs") or self.results_tabs is None:
            return

        # Only refresh live if the user is currently viewing the Losses tab.
        if self.results_tabs.currentWidget() != self.losses_tab:
            return

        self.build_losses_tab()

    def refresh_file_selector_after_result(self, r=None):
        """
        Refresh the visible selector using the existing SelectorListModel system.

        This does not force the user back to the file selector.
        It only rebuilds the model for the selector page the user is currently viewing.
        """

        view_state = getattr(self, "selector_mode", "files")

        if r is not None:
            _, _, finished_file_idx = r["file_key"]
        else:
            finished_file_idx = None

        # User is looking at the top-level file list.
        if view_state == "files":
            self.selector_model.set_items(
                self.active_run.file_list,
                mode="files"
            )
            return

        # User is looking at the profile list for one multi-profile file.
        if view_state == "profiles":
            current_file_idx = getattr(self, "current_file_idx", None)

            # Only refresh this list if the newly finished result belongs to
            # the file currently being viewed.
            if finished_file_idx is None or finished_file_idx == current_file_idx:
                self.selector_model.set_items(
                    self.active_run.file_list[current_file_idx],
                    mode="profiles"
                )

            return

        # User is viewing an actual profile result.
        # Do not change the page or replace the view while they are looking at it.
        if view_state == "result":
            return


    def on_eval_error(self, message: str):
        try:
            if hasattr(self, "worker") and self.worker:
                self.worker.cancel()
        except Exception:
            pass

        try:
            if hasattr(self, "thread") and self.thread:
                self.thread.quit()
                self.thread.wait()
        except Exception:
            pass

        if hasattr(self, "progress_bar"):
            self.progress_bar.hide()
            self.progress_bar.setValue(0)

        self.completed_jobs = 0
        self.total_jobs = 0

        # self.pause_btn.setEnabled(False)
        # self.resume_btn.setEnabled(False)
        # self.cancel_btn.setEnabled(False)

        QMessageBox.critical(
            self,
            "Processing Error",
            f"An error occurred while processing data:\n\n{message}"
        )
  

    def setup_selector_view(self):
        self.selector_view = QListView()
        self.selector_view.setUniformItemSizes(True)
        self.selector_view.setSpacing(2)

        self.selector_model = SelectorListModel([], mode="files", parent=self)
        self.selector_view.setModel(self.selector_model)

        self.selector_view.setStyleSheet("""
            QListView {
                border: none;
                outline: none;
            }

            QListView::item {
                padding: 10px;
                border: 1px solid #888;
                margin: 1px;
            }

            QListView::item:hover {
                background: #eeeeee;
            }

            QListView::item:selected {
                background: #cde8ff;
                color: black;
            }
        """)

        self.selector_view.clicked.connect(self.on_selector_row_clicked)

        self.selector_mode = "files"
        self.current_file_idx = None

    
    def show_file_selector_first_run(self):
        self.clear_layout_detach(self.processing_panel)

        self.proc_panel_title.setText("PROFILE DISPLAY")
        self.processing_panel.addWidget(self.proc_panel_title)
        self.processing_panel.addWidget(self.title_line)
        self.processing_panel.setAlignment(Qt.AlignTop)

        self.loss_graph_btn = QPushButton('Losses over Strike')

        # Only create the stack once
        if not hasattr(self, "main_stack"):
            self.clear_layout_detach(self.main_area_layout)

            self.main_area_layout.setAlignment(Qt.AlignTop)

            self.ensure_eval_controls()
            self.main_area_layout.addWidget(self.progress_bar)

            self.setup_profile_display_area()

        self.show_file_selector()
        self.main_stack.setCurrentWidget(self.selector_page)


    def show_file_selector(self):
        self.selector_mode = "files"
        self.current_file_idx = None

        if self.active_run is None:
            return

        self.clear_layout_detach(self.processing_panel)
        self.processing_panel.addWidget(self.proc_panel_title)
        self.processing_panel.addWidget(self.title_line)
        self.processing_panel.setAlignment(Qt.AlignTop)
        self.processing_panel.addStretch()

        self.proc_panel_title.setText("SELECT PROFILE")

        self.proc_panel_title.setText("SELECT FILE")

        self.selector_model.current_file_idx = None
        self.selector_model.high_loss_files = self.active_run.high_loss_files
        self.selector_model.high_loss_profiles = self.active_run.high_loss_profiles
        self.selector_model.set_items(self.active_run.file_list, mode="files")

        self.main_stack.setCurrentWidget(self.selector_page)
        self.selector_view.show()

    def on_selector_row_clicked(self, index):
        row = index.row()

        if self.selector_mode == "files":
            file_idx = row
            file_item = self.active_run.file_list[file_idx]

            if file_item is None:
                return

            if isinstance(file_item, list):
                self.show_profile_selector(file_idx)
            else:
                self.display_profile(file_idx, None)

        elif self.selector_mode == "profiles":
            profile_idx = row
            file_idx = self.current_file_idx

            if self.active_run.file_list[file_idx][profile_idx] is None:
                return

            self.display_profile(file_idx, profile_idx)


    def show_profile_selector(self, file_idx):
        self.selector_mode = "profiles"
        self.current_file_idx = file_idx

        self.selector_model.current_file_idx = file_idx
        self.selector_model.high_loss_files = self.active_run.high_loss_files
        self.selector_model.high_loss_profiles = self.active_run.high_loss_profiles
        self.selector_model.set_items(self.active_run.file_list[file_idx], mode="profiles")

        self.main_stack.setCurrentWidget(self.selector_page)
        self.selector_view.show()

        self.clear_layout_detach(self.processing_panel)
        self.processing_panel.addWidget(self.proc_panel_title)
        self.processing_panel.addWidget(self.title_line)
        self.processing_panel.setAlignment(Qt.AlignTop)
        self.processing_panel.addStretch()

        self.proc_panel_title.setText("SELECT PROFILE")       

        back_btn = QPushButton("Back")
        back_btn.clicked.connect(self.show_file_selector)
        self.processing_panel.addWidget(back_btn)

        self.processing_panel.setAlignment(Qt.AlignTop)


    def on_back_clicked(self):

        # Restore processing side panel title
        #self.clear_layout_detach(self.processing_panel)

        # Decide which selector to show
        if hasattr(self, "last_file_idx") and self.last_profile_idx is not None:
            self.show_profile_selector(self.last_file_idx)
        else:
            self.show_file_selector()

        # Force the visible page
        self.main_stack.setCurrentWidget(self.selector_page)
        self.selector_view.show()


    def get_adjacent_profile_position(self, file_index, profile_index, direction):
        """
        direction = -1 for previous completed profile
        direction =  1 for next completed profile
        """

        if self.active_run is None:
            return None

        current_key = (file_index, profile_index)

        current_flat_idx = self.active_run.profile_nav_lookup.get(current_key)

        if current_flat_idx is None:
            return None

        i = current_flat_idx + direction

        while 0 <= i < len(self.active_run.profile_nav_positions):
            pos = self.active_run.profile_nav_positions[i]

            if pos is not None:
                return pos

            i += direction

        return None
    

    def go_to_adjacent_profile(self, direction):
        pos = self.get_adjacent_profile_position(
            self.last_file_idx,
            self.last_profile_idx,
            direction
        )

        if pos is None:
            return

        file_idx, profile_idx = pos
        self.display_profile(file_idx, profile_idx)


    def display_profile(self, file_index, profile_index):
        self.last_file_idx = file_index
        self.last_profile_idx = profile_index
        self.selector_mode = "result"

        self.clear_layout_delete(self.profile_page_layout)
        self.clear_layout_detach(self.processing_panel)

        table, tabs, hist_button = self.build_profile_view(file_index, profile_index, flag=True)

        #self.main_area_layout.addWidget(tabs)
        self.profile_page_layout.addWidget(tabs)

        self.proc_panel_title.setText("Parameters for Profile")
        self.processing_panel.addWidget(self.proc_panel_title)
        self.processing_panel.addWidget(self.title_line)
        self.processing_panel.addWidget(table)
        self.processing_panel.addWidget(hist_button)

        nav_layout = QHBoxLayout()

        prev_btn = QPushButton("Previous Figure")
        next_btn = QPushButton("Next Figure")

        prev_pos = self.get_adjacent_profile_position(
        file_index,
        profile_index,
        direction=-1
        )

        next_pos = self.get_adjacent_profile_position(
            file_index,
            profile_index,
            direction=1
        )

        prev_btn.setEnabled(prev_pos is not None)
        next_btn.setEnabled(next_pos is not None)

        prev_btn.clicked.connect(lambda: self.go_to_adjacent_profile(-1))
        next_btn.clicked.connect(lambda: self.go_to_adjacent_profile(1))

        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(next_btn)
        self.processing_panel.addLayout(nav_layout)

        back_btn = QPushButton("Back to Profile List")
        back_btn.clicked.connect(self.on_back_clicked)
        self.processing_panel.addWidget(back_btn)

        self.main_stack.setCurrentWidget(self.profile_page)


    def build_profile_view(self, file_index, profile_index, flag=False):
        if profile_index is None:
            profile = self.active_run.file_list[file_index]
        else:
            profile = self.active_run.file_list[file_index][profile_index]

        fig_path, df, u, fig_loss_path = profile['fig'], profile['table'], profile['uncert'], profile['losses']
        file_info = profile['file_info']

        # if flag:
        #     self.prof_fig = fig

        tabs = QTabWidget()

        plot_label = ScaledImageLabel(fig_path)
        tab1 = QWidget()
        tab1_layout = QVBoxLayout()
        tab1_layout.setContentsMargins(0, 0, 0, 0)
        tab1_layout.addWidget(plot_label)
        tab1.setLayout(tab1_layout)
        tabs.addTab(tab1, "Profile Plot")

        if u is not None:
            canvas_uncert = FigureCanvas(u[0])
            tab2 = QWidget()
            tab2_layout = QVBoxLayout()
            tab2_layout.addWidget(canvas_uncert)
            tab2.setLayout(tab2_layout)
            tabs.addTab(tab2, "Uncertainty")

        
        loss_label = ScaledImageLabel(fig_loss_path)
        tab3 = QWidget()
        tab3_layout = QVBoxLayout()
        tab3_layout.setContentsMargins(0, 0, 0, 0)
        tab3_layout.addWidget(loss_label)
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

        table.setMinimumWidth(300)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()

        plot_history_button = QPushButton("Show Fit History")
        plot_history_button.clicked.connect(lambda: self.show_fit_history(tabs, file_info))

        return table, tabs, plot_history_button


    def quick_eval_old(self, file_info):
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
