from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTreeWidget, QTreeWidgetItem
)
from PyQt5.QtCore import Qt, QSignalBlocker
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class ProfileMetricGraphWidget(QWidget):
    """
    Generic graph widget for profile-level metrics.

    metric="loss":
        Uses run.loss_graph_vals
        One stacked axis

    metric="disp":
        Uses run.disp_graph_vals
        One stacked axis for 1D
        Two stacked axes for 2D
    """

    def __init__(self, main_window, metric, parent=None):
        super().__init__(parent)

        self.main_window = main_window
        self.metric = metric

        self.updating_tree = False
        print("XVALS ")
        print(self.main_window.active_run.loss_x_vals)

        self.hover_data = None
        self.hover_annotation = None
        self.hover_connected = False

        self.parent_items = {}
        self.child_items = {}

        self.init_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def init_ui(self):
        outer_layout = QHBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Files / Profiles", "Move"])
        self.tree.setColumnWidth(0, 320)
        self.tree.setAlternatingRowColors(True)

        self.tree.itemChanged.connect(self.on_tree_item_changed)
        self.tree.itemExpanded.connect(self.on_item_expanded)
        self.tree.itemCollapsed.connect(self.on_item_collapsed)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel(self.get_selector_title()))
        left_layout.addWidget(self.tree)

        self.fig = Figure(figsize=(9, 5))
        self.axes = []
        self.canvas = FigureCanvas(self.fig)

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.canvas)

        outer_layout.addWidget(left_panel, stretch=1)
        outer_layout.addWidget(right_panel, stretch=3)

    def get_selector_title(self):
        if self.metric == "loss":
            return "Select losses to plot"
        if self.metric == "disp":
            return "Select displacements to plot"
        return "Select profiles to plot"

    # ------------------------------------------------------------------
    # Main refresh entry point
    # ------------------------------------------------------------------

    def refresh(self, rebuild_tree=False, result=None):
        """
        Call this when:
        - active run changes
        - a result arrives
        - tab becomes visible
        - file order changes

        Use rebuild_tree=True only when needed.
        """
        run = self.get_run()
        vals = self.get_metric_vals()

        if run is None or not vals:
            self.show_empty_message()
            return

        if rebuild_tree or self.tree.topLevelItemCount() == 0:
            self.populate_tree()
            self.update_plot()
            return
        
        if result is not None:
            self.refresh_after_result(result)
            return

        self.update_plot()

    def show_empty_message(self):
        self.tree.clear()

        self.fig.clear()
        self.axes = [self.fig.add_subplot(111)]
        ax = self.axes[0]

        ax.text(
            0.5,
            0.5,
            f"No {self.metric} values are available yet.",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )

        ax.set_axis_off()
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Run / metric data access
    # ------------------------------------------------------------------

    def get_run(self):
        return self.main_window.active_run
    
    def get_state(self):
        run = self.get_run()
        if run is None:
            return None

        return run.metric_graph_state[self.metric]

    def get_metric_vals(self):
        run = self.get_run()
        if run is None:
            return []

        if self.metric == "loss":
            return run.loss_graph_vals

        if self.metric == "disp":
            return run.disp_graph_vals

        raise ValueError(f"Unknown metric: {self.metric}")

    def get_metric_title(self):
        if self.metric == "loss":
            return "Final Loss by Selected Profiles"

        if self.metric == "disp":
            return "Displacement by Selected Profiles"

        return "Selected Profile Values"

    def get_ylabels(self, n_axes):
        if self.metric == "loss":
            return ["Final Loss"]

        if self.metric == "disp":
            if n_axes == 1:
                return ["Displacement"]
            return [f"Displacement dim {i + 1}" for i in range(n_axes)]

        return [f"Value {i + 1}" for i in range(n_axes)]

    # ------------------------------------------------------------------
    # Tree state
    # ------------------------------------------------------------------

    def init_tree_state(self):
        run = self.get_run()
        state = self.get_state()
        vals = self.get_metric_vals()

        if run is None or state is None:
            return

        n_files = len(vals)

        if len(state.file_order) != n_files:
            state.file_order = list(range(n_files))

        for file_idx in state.file_order:
            if file_idx not in state.file_checked:
                state.file_checked[file_idx] = True

            if file_idx not in state.first_unchecked_child:
                state.first_unchecked_child[file_idx] = None

        state.file_checked = {
            file_idx: checked
            for file_idx, checked in state.file_checked.items()
            if file_idx in state.file_order
        }

        # for file_idx, file_vals in enumerate(vals):
        #     for profile_idx in range(len(file_vals)):
        #         key = (file_idx, profile_idx)
        #         if key not in state.profile_checked:
        #             state.profile_checked[key] = True

    def populate_tree(self):
        run = self.get_run()
        state = self.get_state()

        if run is None or state is None:
            return

        self.init_tree_state()

        self.updating_tree = True
        self.tree.clear()
        self.parent_items.clear()
        self.child_items.clear()

        n_files = len(state.file_order)

        for row, file_idx in enumerate(state.file_order):
            file_item = run.file_list[file_idx]
            file_name = self.get_file_display_name(file_idx)

            completed = run.completed_vals[file_idx]
            total = len(file_item)

            parent = QTreeWidgetItem([f"{file_name}  ({completed}/{total} complete)", ""])
            parent.setData(0, Qt.UserRole, {
                "kind": "file",
                "file_idx": file_idx,
                "profile_idx": None,
            })

            parent.setFlags(
                parent.flags()
                | Qt.ItemIsUserCheckable
                | Qt.ItemIsEnabled
                | Qt.ItemIsSelectable
            )

            self.tree.addTopLevelItem(parent)
            self.parent_items[file_idx] = parent

            if n_files > 1:
                move_widget = self.make_move_widget(file_idx, row, n_files)
                self.tree.setItemWidget(parent, 1, move_widget)

            for profile_idx in range(len(file_item)):
                self.add_profile_child(parent, file_idx, profile_idx)

            self.update_parent_check_state(parent, file_idx)

            parent.setExpanded(file_idx in state.expanded_files)

        self.updating_tree = False

    def refresh_after_result(self, result):
        """
        Update only the tree row affected by one completed profile result.
        Then redraw the plot.

        result["file_key"] format:
            (file_name, profile_idx, file_idx)
        """
        run = self.get_run()
        vals = self.get_metric_vals()

        if run is None or not vals:
            return

        try:
            _, profile_idx, file_idx = result["file_key"]
        except Exception:
            # If result format is unexpected, fall back to full rebuild.
            self.refresh(rebuild_tree=True)
            return

        self.refresh_profile_tree_item(file_idx, profile_idx)
        self.refresh_parent_tree_item(file_idx)

        self.update_plot()

    def refresh_profile_tree_item(self, file_idx, profile_idx):
        """
        Update the label for a single profile row.
        """
        vals = self.get_metric_vals()

        child = self.child_items.get((file_idx, profile_idx))

        if child is None:
            # Tree might not have this row yet.
            # This should be rare if the tree was built from placeholders.
            return

        label = self.get_profile_label(file_idx, profile_idx)

        try:
            value = vals[file_idx][profile_idx]
        except Exception:
            value = None

        if value is None:
            label += "  Processing..."

        self.updating_tree = True
        child.setText(0, label)
        self.updating_tree = False

    def refresh_parent_tree_item(self, file_idx):
        """
        Update only the parent row text for one file/folder.
        Uses run.completed_vals, so this stays O(1).
        """
        run = self.get_run()

        if run is None:
            return

        parent = self.parent_items.get(file_idx)

        if parent is None:
            return

        try:
            file_item = run.file_list[file_idx]
            file_name = self.get_file_display_name(file_idx)
            completed = run.completed_vals[file_idx]
            total = len(file_item)
            text = f"{file_name}  ({completed}/{total} complete)"
        except Exception:
            text = self.get_file_display_name(file_idx)

        self.updating_tree = True
        parent.setText(0, text)
        self.updating_tree = False

    def refresh_tree_labels_only(self):
        run = self.get_run()
        vals = self.get_metric_vals()

        if run is None:
            return

        self.updating_tree = True

        for row in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(row)
            parent_data = parent.data(0, Qt.UserRole)

            if not parent_data:
                continue

            file_idx = parent_data["file_idx"]
            file_item = run.file_list[file_idx]
            file_name = self.get_file_display_name(file_idx)

            completed = run.completed_vals[file_idx] if file_idx < len(run.completed_vals) else 0
            total = len(file_item)

            parent.setText(0, f"{file_name}  ({completed}/{total} complete)")

            for profile_idx in range(parent.childCount()):
                child = parent.child(profile_idx)
                label = self.get_profile_label(file_idx, profile_idx)

                value = vals[file_idx][profile_idx]
                if value is None:
                    label += "  Processing..."

                child.setText(0, label)

        self.updating_tree = False

    def add_profile_child(self, parent, file_idx, profile_idx):
        state = self.get_state()
        vals = self.get_metric_vals()

        label = self.get_profile_label(file_idx, profile_idx)

        value = vals[file_idx][profile_idx]
        if value is None:
            label += "  Processing..."

        child = QTreeWidgetItem([label, ""])
        child.setData(0, Qt.UserRole, {
            "kind": "profile",
            "file_idx": file_idx,
            "profile_idx": profile_idx,
        })

        child.setFlags(
            child.flags()
            | Qt.ItemIsUserCheckable
            | Qt.ItemIsEnabled
            | Qt.ItemIsSelectable
        )

        checked = state.profile_checked.get((file_idx, profile_idx), True)
        child.setCheckState(0, Qt.Checked if checked else Qt.Unchecked)

        parent.addChild(child)
        self.child_items[(file_idx, profile_idx)] = child

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def get_file_display_name(self, file_idx):
        run = self.get_run()

        if run is None:
            return f"File {file_idx + 1}"

        if hasattr(run, "file_names") and file_idx < len(run.file_names):
            name = run.file_names[file_idx]
            if name is not None:
                return str(name)

        return f"File {file_idx + 1}"

    def get_profile_label(self, file_idx, profile_idx):
        run = self.get_run()

        if run is None:
            return f"Profile {profile_idx + 1}"

        try:
            profile = run.file_list[file_idx][profile_idx]

            if isinstance(profile, dict):
                file_num = str(profile.get("file_num", profile_idx + 1))
                prefix = "Profile " if "rofile" not in file_num else ""
                return prefix + file_num

        except Exception:
            pass

        try:
            return str(run.loss_x_vals[file_idx][profile_idx])
        except Exception:
            pass

        return f"Profile {profile_idx + 1}"

    # ------------------------------------------------------------------
    # Move buttons
    # ------------------------------------------------------------------

    def make_move_widget(self, file_idx, row, n_files):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        up_btn = QPushButton("↑")
        down_btn = QPushButton("↓")

        up_btn.setFixedWidth(28)
        down_btn.setFixedWidth(28)

        up_btn.setEnabled(row > 0)
        down_btn.setEnabled(row < n_files - 1)

        up_btn.clicked.connect(
            lambda _, idx=file_idx: self.move_file(idx, -1)
        )
        down_btn.clicked.connect(
            lambda _, idx=file_idx: self.move_file(idx, 1)
        )

        layout.addWidget(up_btn)
        layout.addWidget(down_btn)

        return widget

    def move_file(self, file_idx, direction):
        state = self.get_state()
        if state is None:
            return

        if file_idx not in state.file_order:
            return

        old_pos = state.file_order.index(file_idx)
        new_pos = old_pos + direction

        if new_pos < 0 or new_pos >= len(state.file_order):
            return

        state.file_order[old_pos], state.file_order[new_pos] = (
            state.file_order[new_pos],
            state.file_order[old_pos],
        )

        self.updating_tree = True
        blocker = QSignalBlocker(self.tree)

        item = self.tree.takeTopLevelItem(old_pos)
        self.tree.insertTopLevelItem(new_pos, item)

        del blocker
        self.updating_tree = False

        self.refresh_move_buttons()
        self.tree.setCurrentItem(item)
        self.tree.scrollToItem(item)

        self.update_plot()

    def refresh_move_buttons(self):
        state = self.get_state()
        if state is None:
            return

        n_files = self.tree.topLevelItemCount()

        for row in range(n_files):
            item = self.tree.topLevelItem(row)
            data = item.data(0, Qt.UserRole)

            if not data:
                continue

            file_idx = data["file_idx"]

            old_widget = self.tree.itemWidget(item, 1)
            if old_widget is not None:
                self.tree.removeItemWidget(item, 1)
                old_widget.deleteLater()

            if n_files > 1:
                move_widget = self.make_move_widget(file_idx, row, n_files)
                self.tree.setItemWidget(item, 1, move_widget)

    # ------------------------------------------------------------------
    # Checkbox behavior
    # ------------------------------------------------------------------

    def on_tree_item_changed(self, item, column):
        if column != 0:
            return

        if self.updating_tree:
            return

        state = self.get_state()
        if state is None:
            return

        data = item.data(0, Qt.UserRole)
        if not data:
            return

        kind = data["kind"]
        file_idx = data["file_idx"]
        profile_idx = data["profile_idx"]

        check_state = item.checkState(0)

        if kind == "file":
            checked = check_state == Qt.Checked
            child_state = Qt.Checked if checked else Qt.Unchecked

            self.updating_tree = True
            blocker = QSignalBlocker(self.tree)

            for i in range(item.childCount()):
                child = item.child(i)
                child_data = child.data(0, Qt.UserRole)

                child.setCheckState(0, child_state)

                if child_data:
                    c_file_idx = child_data["file_idx"]
                    c_profile_idx = child_data["profile_idx"]
                    state.profile_checked[(c_file_idx, c_profile_idx)] = checked

            state.first_unchecked_child[file_idx] = None if checked else 0

            del blocker
            self.updating_tree = False

        elif kind == "profile":
            checked = check_state == Qt.Checked
            state.profile_checked[(file_idx, profile_idx)] = checked

            parent = item.parent()
            if parent is not None:
                parent_data = parent.data(0, Qt.UserRole)
                if parent_data:
                    parent_file_idx = parent_data["file_idx"]
                    self.update_parent_check_state(parent, parent_file_idx)

        self.update_plot()

    def update_parent_check_state(self, parent, file_idx):
        if parent.childCount() == 0:
            return

        state = self.get_state()
        if state is None:
            return

        remembered_child = state.first_unchecked_child.get(file_idx)

        parent_should_be_checked = True

        if remembered_child is not None and remembered_child < parent.childCount():
            if parent.child(remembered_child).checkState(0) == Qt.Unchecked:
                parent_should_be_checked = False
            else:
                remembered_child = None

        if remembered_child is None:
            for i in range(parent.childCount()):
                if parent.child(i).checkState(0) == Qt.Unchecked:
                    state.first_unchecked_child[file_idx] = i
                    parent_should_be_checked = False
                    break
            else:
                state.first_unchecked_child[file_idx] = None
                parent_should_be_checked = True

        new_state = Qt.Checked if parent_should_be_checked else Qt.Unchecked

        if parent.checkState(0) != new_state:
            blocker = QSignalBlocker(self.tree)
            parent.setCheckState(0, new_state)
            del blocker

    def on_item_expanded(self, item):
        state = self.get_state()
        if state is None:
            return

        data = item.data(0, Qt.UserRole)
        if data and data["kind"] == "file":
            state.expanded_files.add(data["file_idx"])


    def on_item_collapsed(self, item):
        state = self.get_state()
        if state is None:
            return

        data = item.data(0, Qt.UserRole)
        if data and data["kind"] == "file":
            state.expanded_files.discard(data["file_idx"])

    # ------------------------------------------------------------------
    # Plot data
    # ------------------------------------------------------------------

    def normalize_value_to_series(self, value):
        if value is None:
            return None

        if isinstance(value, np.ndarray):
            value = value.tolist()

        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]

        return [float(value)]

    def get_selected_plot_data(self):
        run = self.get_run()
        state = self.get_state()
        vals = self.get_metric_vals()

        if run is None or state is None:
            return [], [], []

        x_vals = []
        y_series = []
        hover_labels = []

        for file_idx in state.file_order:
            file_name = self.get_file_display_name(file_idx)

            for profile_idx, value in enumerate(vals[file_idx]):
                if not state.profile_checked.get((file_idx, profile_idx), True):
                    continue

                series_values = self.normalize_value_to_series(value)

                if series_values is None:
                    continue

                while len(y_series) < len(series_values):
                    y_series.append([])

                for dim_idx, y in enumerate(series_values):
                    y_series[dim_idx].append(y)

                profile_label = self.get_profile_label(file_idx, profile_idx)

                if len(series_values) == 1:
                    value_text = f"value = {series_values[0]:.6g}"
                else:
                    value_text = "\n".join(
                        f"dim {i + 1} = {v:.6g}"
                        for i, v in enumerate(series_values)
                    )
                x = run.loss_x_vals[file_idx][profile_idx]
                x_vals.append(x)
                hover_labels.append(
                    f"{file_name}\n{profile_label}\nx = {x}\n{value_text}"
                )


        return x_vals, y_series, hover_labels

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def update_plot(self):
        x_vals, y_series, hover_labels = self.get_selected_plot_data()

        self.fig.clear()
        self.axes = []

        if not y_series:
            ax = self.fig.add_subplot(111)
            self.axes.append(ax)

            ax.text(
                0.5,
                0.5,
                "No selected completed profile values",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()

            self.hover_data = None
            self.canvas.draw_idle()
            return

        n_axes = len(y_series)
        ylabels = self.get_ylabels(n_axes)

        for dim_idx, y_vals in enumerate(y_series):
            ax = self.fig.add_subplot(n_axes, 1, dim_idx + 1)
            self.axes.append(ax)

            ax.plot(x_vals, y_vals, marker="o", linestyle="-", picker=5)

            ylabel = ylabels[dim_idx] if dim_idx < len(ylabels) else f"Value {dim_idx + 1}"
            self.format_axis(ax, x_vals, ylabel)

            if dim_idx != n_axes - 1:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

        self.axes[0].set_title(self.get_metric_title())

        self.hover_data = {
            "x": x_vals,
            "series": y_series,
            "labels": hover_labels,
        }

        self.connect_hover_once()

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def format_axis(self, ax, x_vals, ylabel):
        ax.set_xlabel("Selected profile index")
        ax.set_ylabel(ylabel)
        ax.grid(True)

        if not x_vals:
            return

        n = len(x_vals)

        if n <= 30:
            step = 1
        elif n <= 100:
            step = 5
        elif n <= 500:
            step = 25
        elif n <= 2000:
            step = 100
        else:
            step = 250

        ticks = list(range(1, n + 1, step))

        if ticks[-1] != n:
            ticks.append(n)

        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])

    # ------------------------------------------------------------------
    # Hover behavior
    # ------------------------------------------------------------------

    def connect_hover_once(self):
        if self.hover_connected:
            return

        self.hover_connected = True
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)

    def on_hover(self, event):
        if event.inaxes not in self.axes:
            self.hide_hover()
            return

        data = self.hover_data

        if not data:
            self.hide_hover()
            return

        x_vals = data["x"]
        y_series = data["series"]
        labels = data["labels"]

        if not x_vals or event.xdata is None or event.ydata is None:
            self.hide_hover()
            return

        idx = int(round(event.xdata)) - 1

        if idx < 0 or idx >= len(x_vals):
            self.hide_hover()
            return

        ax = event.inaxes
        axis_idx = self.axes.index(ax)

        if axis_idx >= len(y_series):
            self.hide_hover()
            return

        x = x_vals[idx]
        y = y_series[axis_idx][idx]

        y_vals = y_series[axis_idx]
        y_range = max(y_vals) - min(y_vals) if len(y_vals) > 1 else 1.0

        if abs(event.ydata - y) > 0.05 * y_range:
            self.hide_hover()
            return

        if self.hover_annotation is None or self.hover_annotation.axes is not ax:
            if self.hover_annotation is not None:
                try:
                    self.hover_annotation.remove()
                except Exception:
                    pass

            self.hover_annotation = ax.annotate(
                "",
                xy=(0, 0),
                xytext=(15, 15),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"),
                arrowprops=dict(arrowstyle="->"),
            )

        self.hover_annotation.xy = (x, y)
        self.hover_annotation.set_text(labels[idx])
        self.hover_annotation.set_visible(True)
        self.canvas.draw_idle()

    def hide_hover(self):
        if self.hover_annotation is not None and self.hover_annotation.get_visible():
            self.hover_annotation.set_visible(False)
            self.canvas.draw_idle()