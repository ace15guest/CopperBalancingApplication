from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QSplitter, QLabel, QComboBox,
)
from PyQt6.QtCore import Qt

from ui.components.stackup_editor import StackupEditor
from ui.components.stackup_diagram import StackupDiagram
from src.models import Stackup
from src.ingestion.materials import load_material_sets


class StackupPage(QWidget):
    def __init__(self):
        super().__init__()
        self._build()

    def _build(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # --- Left: cross-section diagram ---
        self._diagram = StackupDiagram()
        self._diagram.setFixedWidth(180)
        splitter.addWidget(self._diagram)

        # --- Right: toolbar + table ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(6)

        # --- Material set selector ---
        mat_bar = QHBoxLayout()
        mat_bar.setSpacing(6)
        mat_bar.addWidget(QLabel("Material Set:"))
        self._material_combo = QComboBox()
        self._material_combo.setFixedWidth(140)
        self._material_sets = load_material_sets()
        self._material_combo.addItems([ms.name for ms in self._material_sets])
        mat_bar.addWidget(self._material_combo)
        mat_bar.addStretch()
        right_layout.addLayout(mat_bar)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        add_copper_btn = QPushButton("Add Copper Layer")
        add_dielectric_btn = QPushButton("Add Dielectric")
        delete_btn = QPushButton("Delete Row")
        up_btn = QPushButton("Move Up")
        down_btn = QPushButton("Move Down")

        for btn in (add_copper_btn, add_dielectric_btn, delete_btn, up_btn, down_btn):
            toolbar.addWidget(btn)
        toolbar.addStretch()

        self._editor = StackupEditor()
        # Initialise and keep in sync with the material-set selector
        self._editor.set_material_set(self._material_combo.currentText())
        self._material_combo.currentTextChanged.connect(self._editor.set_material_set)

        add_copper_btn.clicked.connect(self._editor.add_copper_row)
        add_dielectric_btn.clicked.connect(self._editor.add_dielectric_row)
        delete_btn.clicked.connect(self._editor.delete_selected_row)
        up_btn.clicked.connect(self._editor.move_selected_up)
        down_btn.clicked.connect(self._editor.move_selected_down)
        self._editor.stackup_changed.connect(self._refresh_diagram)

        right_layout.addLayout(toolbar)
        right_layout.addWidget(self._editor)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setHandleWidth(4)

        layout.addWidget(splitter)

    def _refresh_diagram(self):
        stackup = self._editor.get_stackup()
        self._diagram.update_rows(stackup.rows)

    def get_stackup(self) -> Stackup:
        stackup = self._editor.get_stackup()
        stackup.material_set_name = self._material_combo.currentText()
        return stackup

    def load_stackup(self, stackup: Stackup) -> None:
        self._editor.load_rows(stackup.rows)
        if stackup.material_set_name in [ms.name for ms in self._material_sets]:
            self._material_combo.setCurrentText(stackup.material_set_name)

    def new_stackup(self) -> None:
        self._editor.clear()
