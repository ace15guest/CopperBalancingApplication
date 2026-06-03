from PyQt6.QtWidgets import QLabel
from PyQt6.QtCore import pyqtSignal
from pathlib import Path


class FileDropzone(QLabel):
    """Drag-and-drop or browse widget that emits the selected file path."""

    file_selected = pyqtSignal(Path)

    def __init__(self, label: str = "Drop file here or click to browse"):
        super().__init__(label)
        self.setAcceptDrops(True)
