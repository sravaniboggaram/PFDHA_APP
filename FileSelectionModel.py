from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex
from PyQt5.QtGui import QFont


class SelectorListModel(QAbstractListModel):
    def __init__(self, items, mode="files", parent=None):
        super().__init__(parent)
        self.items = items
        self.mode = mode

    def rowCount(self, parent=QModelIndex()):
        return len(self.items)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        item = self.items[row]

        if role == Qt.DisplayRole:
            if self.mode == "files":
                # Top-level self.file_list
                if isinstance(item, list):
                    # File containing multiple profiles
                    return item[0]["file_key"][0]
                else:
                    # Single profile file
                    return item["file_key"][0]

            elif self.mode == "profiles":
                # Profile inside a multi-profile file
                print("ITEM: ", item)
                return "Profile " + item["file_num"]

        if role == Qt.FontRole:
            if self.mode == "files" and isinstance(item, list):
                font = QFont()
                font.setBold(True)
                return font

        return None