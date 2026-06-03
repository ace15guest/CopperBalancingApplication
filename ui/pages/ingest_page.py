from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QSplitter, QFileDialog,
)
from PyQt6.QtCore import Qt
from ui.components.color_scale_editor import ColorScaleEditor
from ui.components.heatmap_view import HeatmapView


class IngestPage(QWidget):
    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- Top bar ---
        top = QHBoxLayout()
        self._folder_path = QLineEdit()
        self._folder_path.setPlaceholderText("Gerber Folder Path")
        self._folder_path.setReadOnly(True)
        top.addWidget(self._folder_path)

        browse_btn = QPushButton("Gerber Folder")
        browse_btn.setFixedWidth(120)
        browse_btn.clicked.connect(self._browse)
        top.addWidget(browse_btn)
        layout.addLayout(top)

        # --- Splitter ---
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._color_editor = ColorScaleEditor()
        self._color_editor.setFixedWidth(210)
        splitter.addWidget(self._color_editor)

        self._heatmap = HeatmapView()
        splitter.addWidget(self._heatmap)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setHandleWidth(4)

        layout.addWidget(splitter)

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Gerber Folder")
        if folder:
            self._folder_path.setText(folder)
