from PyQt5.QtWidgets import (
    QLabel, QDialog, QComboBox, QDialogButtonBox, QFormLayout,
    QHBoxLayout, QRadioButton, QLineEdit, QButtonGroup,
    QVBoxLayout
)


class SelectH5Subset(QDialog):
    def __init__(self, num_files):
        super().__init__()
        self.setWindowTitle("Select H5 File Subset")
        self.setMinimumSize(300, 250)

        self.label = QLabel("Select Profiles for Processing")

        all_layout = QHBoxLayout()
        radio_all = QRadioButton()
        label_all = QLabel("All Profiles")
        label_all.setWordWrap(True)
        label_all.setStyleSheet("margin-left: 5px;")
        all_layout.addWidget(radio_all)
        all_layout.addWidget(label_all)

        subset_layout = QHBoxLayout()
        radio_subset = QRadioButton()
        label_subset = QLabel("Subset")
        label_subset.setWordWrap(True)
        label_subset.setStyleSheet("margin-left: 5px;")
        info_subset = QLabel("ⓘ")
        info_subset.setToolTip("Single profiles or profile ranges separated with commas, ranges with hyphens. Ex: profile_010, profile_190-profile_210")
        info_subset.setWordWrap(True)
        info_subset.setStyleSheet("color: black; margin-left: 5px;")
        subset_layout.addWidget(radio_subset)
        subset_layout.addWidget(label_subset)
        subset_layout.addWidget(info_subset)

        self.text_boxes = []

        for i in range(num_files):
            t = QLineEdit()
            t.setPlaceholderText(f"File {i}: ")
            t.setEnabled(False)
            self.text_boxes.append(t)

        def enable_text():
            if radio_subset.isChecked():
                for text in self.text_boxes:
                    text.setEnabled(True)
            if radio_all.isChecked():
                for text in self.text_boxes:
                    text.setEnabled(False)

        self.group = QButtonGroup(self)
        self.group.addButton(radio_all, id=0)
        self.group.addButton(radio_subset, id=1)
        radio_all.setChecked(True)
        self.group.buttonClicked.connect(enable_text)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(all_layout)
        layout.addLayout(subset_layout)

        for t in self.text_boxes:
            layout.addWidget(t)

        layout.addWidget(self.button_box)
        self.setLayout(layout)

    def selected_options(self):
        return [text_box.text() for text_box in self.text_boxes]