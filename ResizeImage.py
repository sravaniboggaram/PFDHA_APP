from PyQt5.QtWidgets import QLabel, QSizePolicy
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class ScaledImageLabel(QLabel):
    def __init__(self, image_path=None, parent=None):
        super().__init__(parent)

        self._pixmap_original = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(1, 1)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        if image_path is not None:
            self.set_image(image_path)

    def set_image(self, image_path):
        self._pixmap_original = QPixmap(str(image_path))
        self._update_scaled_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._pixmap_original is None or self._pixmap_original.isNull():
            return

        scaled = self._pixmap_original.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )

        self.setPixmap(scaled)