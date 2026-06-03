from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QCheckBox, QSpinBox, QComboBox, QLineEdit, QColorDialog,
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import pyqtSignal


def _even_stops(n: int = 10) -> list[int]:
    """Generate n evenly spaced thresholds from 100 down to 0."""
    return [round(100 * i / (n - 1)) for i in range(n - 1, -1, -1)]


_HEAT2 = [
    QColor("#FF0000"),
    QColor("#FF6B6B"),
    QColor("#FFB3B3"),
    QColor("#FFD9D9"),
    QColor("#FFFFFF"),
    QColor("#D9E5FF"),
    QColor("#99BBFF"),
    QColor("#4488FF"),
    QColor("#0044FF"),
    QColor("#0000CC"),
]

_PRESETS: dict[str, list[QColor]] = {
    "Heat2": _HEAT2,
}


class _ColorSwatch(QPushButton):
    color_changed = pyqtSignal(QColor)

    def __init__(self, color: QColor, parent=None):
        super().__init__(parent)
        self.setFixedSize(36, 22)
        self._apply(color)
        self.clicked.connect(self._pick)

    def _apply(self, color: QColor):
        self._color = color
        self.setStyleSheet(
            f"background-color: {color.name()};"
            f"border: 1px solid #666;"
            f"border-radius: 2px;"
        )

    def _pick(self):
        picked = QColorDialog.getColor(self._color, self)
        if picked.isValid():
            self._apply(picked)
            self.color_changed.emit(picked)

    def color(self) -> QColor:
        return self._color

    def set_color(self, color: QColor):
        self._apply(color)


class ColorScaleEditor(QWidget):
    """
    Editable 10-stop color scale. Each row has a threshold spinbox,
    a clickable color swatch, and a visibility checkbox.
    """
    scale_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[tuple[QSpinBox, _ColorSwatch, QCheckBox]] = []
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(3)

        defaults = _even_stops()
        for i, threshold in enumerate(defaults):
            row = QHBoxLayout()
            row.setSpacing(6)

            spinbox = QSpinBox()
            spinbox.setRange(0, 100)
            spinbox.setValue(threshold)
            spinbox.setSuffix("%")
            spinbox.setFixedWidth(78)
            spinbox.valueChanged.connect(self.scale_changed.emit)

            swatch = _ColorSwatch(_HEAT2[i])
            swatch.color_changed.connect(lambda _: self.scale_changed.emit())

            checkbox = QCheckBox()
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.scale_changed.emit)

            row.addWidget(spinbox)
            row.addWidget(swatch)
            row.addWidget(checkbox)
            layout.addLayout(row)
            self._rows.append((spinbox, swatch, checkbox))

        layout.addSpacing(10)

        self._palette_combo = QComboBox()
        self._palette_combo.addItems(list(_PRESETS.keys()))
        layout.addWidget(self._palette_combo)

        recolor_btn = QPushButton("Recolor")
        recolor_btn.clicked.connect(self._apply_preset)
        layout.addWidget(recolor_btn)

        layout.addSpacing(6)

        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("New Color Palette Name")
        layout.addWidget(self._name_input)

        save_btn = QPushButton("Save Color Palette")
        save_btn.clicked.connect(self._save_palette)
        layout.addWidget(save_btn)

        delete_btn = QPushButton("Delete Color Palette")
        delete_btn.clicked.connect(self._delete_palette)
        layout.addWidget(delete_btn)

        layout.addStretch()

    def _apply_preset(self):
        colors = _PRESETS.get(self._palette_combo.currentText(), [])
        for i, (_, swatch, _cb) in enumerate(self._rows):
            if i < len(colors):
                swatch.set_color(colors[i])
        self.scale_changed.emit()

    def _save_palette(self):
        name = self._name_input.text().strip()
        if not name:
            return
        _PRESETS[name] = [swatch.color() for _, swatch, _ in self._rows]
        if self._palette_combo.findText(name) == -1:
            self._palette_combo.addItem(name)
        self._name_input.clear()

    def _delete_palette(self):
        name = self._palette_combo.currentText()
        if name in _PRESETS and len(_PRESETS) > 1:
            del _PRESETS[name]
            self._palette_combo.removeItem(self._palette_combo.currentIndex())

    def get_scale(self) -> list[tuple[int, QColor, bool]]:
        """Returns (threshold %, QColor, visible) sorted descending by threshold."""
        return sorted(
            [(sb.value(), sw.color(), cb.isChecked()) for sb, sw, cb in self._rows],
            key=lambda x: x[0],
            reverse=True,
        )
