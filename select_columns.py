from PyQt5.QtWidgets import (
    QLabel, QDialog, QComboBox, QDialogButtonBox, QFormLayout
)


class SelectColumns(QDialog):
    def __init__(self, column_names, num_dimensions, locations, loc_format):
        super().__init__()
        self.setWindowTitle("Select Data Columns")
        self.setMinimumSize(300, 250)
        self.locations = locations

        self.label = QLabel("Define the columns in your dataset:")

        # Create dropdowns for Strike and Parallel
        self.combo_strike = QComboBox()
        self.combo_parallel = QComboBox()
        self.combo_prof_ids = QComboBox()
        
        self.combo_strike.addItems(column_names)
        self.combo_parallel.addItems(column_names)
        self.combo_prof_ids.addItem("None")
        self.combo_prof_ids.addItems(column_names)
        
        if num_dimensions == 2:
            self.combo_perpendicular = QComboBox()
            self.combo_perpendicular.addItems(column_names)
        else:
            self.combo_perpendicular = None

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QFormLayout()
        layout.addRow(self.label)
        layout.addRow("Select Distance Along Strike:", self.combo_strike)
        layout.addRow("Select Parallel Displacement:", self.combo_parallel)

        if self.combo_perpendicular:
            layout.addRow("Select Perpendicular Displacement:", self.combo_perpendicular)
        layout.addRow("Profile ID:", self.combo_prof_ids)

        if locations:
            self.combo_point1 = QComboBox()
            self.combo_point2 = QComboBox()
            self.combo_az = QComboBox()
            self.combo_point1.addItem("None")
            self.combo_point1.addItems(column_names)
            self.combo_point2.addItem("None")
            self.combo_point2.addItems(column_names)
            self.combo_az.addItem("None")
            self.combo_az.addItems(column_names)
            
            (text1, text2) = ("Latitude:", "Longitude:") if loc_format == 'LATLON' else ("Northing:", "Easting:")
            layout.addRow(text1, self.combo_point1)
            layout.addRow(text2, self.combo_point2)
            layout.addRow("Azimuth:", self.combo_az)

        layout.addRow(self.button_box)
        self.setLayout(layout)

    def selected_options(self):
        option_perpendicular = self.combo_perpendicular.currentText() if self.combo_perpendicular else None
        if self.locations:
            point1 = self.combo_point1.currentText()
            point2 = self.combo_point2.currentText()
            az = self.combo_az.currentText()
        else:
            point1 = point2 = az = None

        return self.combo_strike.currentText(), self.combo_parallel.currentText(), option_perpendicular, self.combo_prof_ids.currentText(), point1, point2, az