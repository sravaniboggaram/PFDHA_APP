from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex
from PyQt5.QtGui import QFont


class SelectorListModel(QAbstractListModel):
    def __init__(self, items, file_names=None, completed=None, mode="files", parent=None):
        super().__init__(parent)
        self.items = items if items is not None else []
        self.mode = mode
        self.completed = completed
        self.file_names = file_names

        self.high_loss_files = set()
        self.high_loss_profiles = set()
        self.current_file_idx = None

    def set_items(self, items, completed=None, mode=None):
        self.beginResetModel()

        self.items = items
        self.completed = completed
        if mode is not None:
            self.mode = mode

        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return len(self.items)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        item = self.items[row]
        prefix = "  "
        completed = 0 if self.completed is None else self.completed[row]

        if role == Qt.DisplayRole:
            if self.mode == "files":
                if row in self.high_loss_files:
                    prefix = "🔴 "

                file_name = self.file_names[row]

                if item is None:
                    return prefix + file_name + " Processing..."

                total = len(item)
                return prefix + f"{file_name}  ({completed}/{total} profiles complete)"


            elif self.mode == "profiles":
                if (self.current_file_idx, row) in self.high_loss_profiles:
                    prefix = "🔴 "
                if item is None:
                    return prefix + f"Profile {row}  Processing..."
                
                if "rofile" not in str(item["file_num"]):
                    prefix += "Profile "

                return prefix + str(item["file_num"])

        if role == Qt.FontRole:
            if self.mode == "files" and isinstance(item, list):
                font = QFont()
                font.setBold(True)
                return font

        return None