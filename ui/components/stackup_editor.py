from dataclasses import asdict
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QTableWidget, QHeaderView,
    QDoubleSpinBox, QComboBox, QLabel, QPushButton,
    QHBoxLayout, QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

from src.ingestion.materials import load_material_sets
from src.models import Stackup, StackupRow
from ui.components.stackup_diagram import MATERIAL_COLORS, COPPER_COLOR

# ---------------------------------------------------------------------------
# Core / prepreg detection (suffix-based for new materials; legacy set for old)
# ---------------------------------------------------------------------------

_LEGACY_PREPREG = {"PrePreg 1080", "PrePreg 1651", "PrePreg 3080"}


def _is_core(material: str | None) -> bool:
    return bool(material) and (material == "FR4 Core" or material.endswith(" Core"))


def _is_prepreg(material: str | None) -> bool:
    return bool(material) and (material in _LEGACY_PREPREG or material.endswith(" Prepreg"))


def _material_hex(material: str) -> str:
    """Return the display colour for a dielectric material name."""
    c = MATERIAL_COLORS.get(material)
    if c:
        return c
    if _is_core(material):
        return MATERIAL_COLORS.get("FR4 Core", "#8CA85A")
    if _is_prepreg(material):
        return MATERIAL_COLORS.get("PrePreg 1080", "#C8D890")
    return "#888888"


# ---------------------------------------------------------------------------
# Material list — legacy entries + names loaded from materials.json
# ---------------------------------------------------------------------------

def _load_dynamic_materials() -> list[str]:
    """Return dielectric material names from materials.json (excludes FR4 / Copper)."""
    _skip = {"FR4", "Copper"}
    try:
        seen: set[str] = set()
        names: list[str] = []
        for ms in load_material_sets():
            for mat in ms.materials:
                if mat.name not in _skip and mat.name not in seen:
                    seen.add(mat.name)
                    names.append(mat.name)
        return names
    except Exception:
        return []


_DYNAMIC_NAMES = _load_dynamic_materials()

MATERIALS: list[str] = (
    ["Liquid PhotoImageable Mask"]
    + [f"{n} Core"    for n in _DYNAMIC_NAMES]
    + [f"{n} Prepreg" for n in _DYNAMIC_NAMES]
    # Legacy FR4 family kept for backward compatibility with saved stackup files
    + ["PrePreg 1080", "PrePreg 1651", "PrePreg 3080", "FR4 Core"]
)

# Copper type options: label → thickness in mils
# Oz weights apply when at least one adjacent dielectric is core.
_OZ_TYPES: dict[str, float] = {
    "1/2 oz": 0.70,
    "1 oz":   1.40,
    "2 oz":   2.80,
    "3 oz":   4.20,
}
_FOIL_PLATING_TYPES: dict[str, float] = {
    f"foil + plating {v / 10:.1f}": round(v / 10, 1)
    for v in range(10, 41)
}
_ALL_COPPER_TYPES = {**_OZ_TYPES, **_FOIL_PLATING_TYPES}

_DEFAULT_THICKNESS: dict[str, float] = {
    "Liquid PhotoImageable Mask": 1.0,
    "PrePreg 1080":               3.0,
    "PrePreg 1651":               6.0,
    "PrePreg 3080":               3.0,
    "FR4 Core":                   12.0,
    **{f"{n} Core":    12.0 for n in _DYNAMIC_NAMES},
    **{f"{n} Prepreg":  3.0 for n in _DYNAMIC_NAMES},
}

# Legacy material sets that use the FR4 / PrePreg family
_LEGACY_SET_NAMES = {"Set A", "Set B"}
_LEGACY_DIELECTRICS = ["PrePreg 1080", "PrePreg 1651", "PrePreg 3080", "FR4 Core"]


def _get_layer_type(material: str) -> str:
    """Return 'Core', 'Prepreg', or 'Mask' for a full material name."""
    if material == "Liquid PhotoImageable Mask":
        return "Mask"
    if _is_core(material):
        return "Core"
    if _is_prepreg(material):
        return "Prepreg"
    return "Core"


def _compose_material(set_name: str, type_str: str) -> str:
    """Compose the full material name from an active set + layer type string."""
    if type_str == "Mask":
        return "Liquid PhotoImageable Mask"
    if set_name in _LEGACY_SET_NAMES or set_name == "" or set_name not in _DYNAMIC_NAMES:
        return "FR4 Core" if type_str == "Core" else "PrePreg 1080"
    return f"{set_name} {type_str}"

# Standard PCB glass constructions (style — resin%).  Placeholder list;
# material-specific offerings can be added later.
_GLASS_CONSTRUCTIONS = [
    "106 — 76%",
    "1080 — 65%",
    "1080 — 76%",
    "2113 — 55%",
    "2116 — 50%",
    "1652 — 55%",
    "3313 — 46%",
    "7628 — 42%",
]

_COL_LAYER  = 0
_COL_DESC   = 1
_COL_THICK  = 2
_COL_GERBER = 3
_ROW_H      = 38


def _spinbox_style(color: QColor) -> str:
    bg = color.name()
    btn = color.darker(115).name()
    return (
        f"QDoubleSpinBox {{ background-color: {bg}; color: #111; }}"
        f"QDoubleSpinBox::up-button {{ subcontrol-origin: border; subcontrol-position: top right;"
        f" width: 16px; border-left: 1px solid #3d3d3d; border-bottom: 1px solid #3d3d3d;"
        f" background-color: {btn}; border-top-right-radius: 2px; }}"
        f"QDoubleSpinBox::up-button:hover {{ background-color: {color.darker(105).name()}; }}"
        f"QDoubleSpinBox::down-button {{ subcontrol-origin: border; subcontrol-position: bottom right;"
        f" width: 16px; border-left: 1px solid #3d3d3d; background-color: {btn};"
        f" border-bottom-right-radius: 2px; }}"
        f"QDoubleSpinBox::down-button:hover {{ background-color: {color.darker(105).name()}; }}"
        f"QDoubleSpinBox::up-arrow {{ width: 8px; height: 5px; }}"
        f"QDoubleSpinBox::down-arrow {{ width: 8px; height: 5px; }}"
    )


class _GerberCell(QWidget):
    """Compact browse button + filename label for a copper row's Gerber file."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._path: Path | None = None
        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)

        self._label = QLabel("No file")
        self._label.setStyleSheet("color: #888; font-style: italic;")
        self._label.setToolTip("")

        btn = QPushButton("Browse")
        btn.setFixedWidth(60)
        btn.clicked.connect(self._browse)

        layout.addWidget(self._label, stretch=1)
        layout.addWidget(btn)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Gerber File", "",
            "Gerber Files (*.gbr *.ger *.gtl *.gbl *.gts *.gbs *.gto *.gbo *.gtp *.gbp);;All Files (*)",
        )
        if path:
            self.set_path(Path(path))

    def set_path(self, path: Path | None):
        self._path = path
        if path:
            self._label.setText(path.name)
            self._label.setToolTip(str(path))
            self._label.setStyleSheet("")
        else:
            self._label.setText("No file")
            self._label.setToolTip("")
            self._label.setStyleSheet("color: #888; font-style: italic;")

    @property
    def gerber_path(self) -> Path | None:
        return self._path


class StackupEditor(QTableWidget):
    """
    Table for defining the PCB stackup row by row.
    Columns: Layer | Description | Finish Thickness (mils) | Gerber File

    Copper rows show oz weights when both adjacent dielectrics are FR4 Core,
    otherwise foil + plating options.
    """

    stackup_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(0, 4, parent)
        self._suppress = False
        self._active_material_set: str = ""
        self._setup_headers()

    def _setup_headers(self):
        self.setHorizontalHeaderLabels(["Layer", "Description", "Thickness (mils)", "Gerber File"])
        self.horizontalHeader().setSectionResizeMode(_COL_LAYER,  QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().setSectionResizeMode(_COL_DESC,   QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(_COL_THICK,  QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().setSectionResizeMode(_COL_GERBER, QHeaderView.ResizeMode.Stretch)
        self.setColumnWidth(_COL_LAYER, 65)
        self.setColumnWidth(_COL_THICK, 150)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_copper_row(self):
        self._insert_row({"row_type": "copper", "layer_number": None,
                          "material": "", "finish_thickness_mil": 1.40,
                          "gerber_path": None})

    def add_dielectric_row(self):
        mat = "FR4 Core"
        self._insert_row({"row_type": "dielectric", "layer_number": None,
                          "material": mat, "finish_thickness_mil": _DEFAULT_THICKNESS[mat],
                          "gerber_path": None})

    def delete_selected_row(self):
        row = self.currentRow()
        if row < 0:
            return
        self.removeRow(row)
        self._renumber()
        self.stackup_changed.emit()

    def move_selected_up(self):
        row = self.currentRow()
        if row <= 0:
            return
        self._swap(row - 1, row)
        self.setCurrentCell(row - 1, 0)
        self.stackup_changed.emit()

    def move_selected_down(self):
        row = self.currentRow()
        if row < 0 or row >= self.rowCount() - 1:
            return
        self._swap(row, row + 1)
        self.setCurrentCell(row + 1, 0)
        self.stackup_changed.emit()

    def set_material_set(self, set_name: str) -> None:
        """Update the active material set and refresh all dielectric rows."""
        self._active_material_set = set_name
        for i in range(self.rowCount()):
            if not self._is_copper_row(i):
                btn = self.cellWidget(i, _COL_LAYER)
                if isinstance(btn, QPushButton):
                    self._on_type_changed(i, btn.text())
        self._refresh_material_options()

    def get_stackup(self) -> Stackup:
        rows = [self._extract(i) for i in range(self.rowCount())]
        return Stackup(rows=rows)

    def load_rows(self, rows: list[StackupRow]) -> None:
        self.setRowCount(0)
        for row in rows:
            r = self.rowCount()
            self.insertRow(r)
            self.setRowHeight(r, _ROW_H)
            self._populate(r, asdict(row))
        self._renumber()
        self.stackup_changed.emit()

    def clear(self) -> None:
        self.setRowCount(0)
        self.stackup_changed.emit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_copper_row(self, row: int) -> bool:
        return isinstance(self.cellWidget(row, _COL_GERBER), _GerberCell)

    def _insert_row(self, data: dict):
        row = self.rowCount()
        self.insertRow(row)
        self.setRowHeight(row, _ROW_H)
        self._populate(row, data)
        self._renumber()
        self.stackup_changed.emit()

    def _populate(self, row: int, data: dict):
        is_copper = data["row_type"] == "copper"
        color = QColor(COPPER_COLOR if is_copper else _material_hex(data["material"]))

        if is_copper:
            # Col 0 — Layer number label (populated by _renumber)
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"background-color: {color.name()}; color: #111;")
            self.setCellWidget(row, _COL_LAYER, lbl)

            # Col 1 — Copper type combo
            combo = QComboBox()
            combo.setStyleSheet(f"background-color: {color.name()}; color: #111;")
            combo.blockSignals(True)
            combo.addItems(list(_ALL_COPPER_TYPES.keys()))
            if data["material"] in _ALL_COPPER_TYPES:
                combo.setCurrentText(data["material"])
            combo.blockSignals(False)
            combo.currentTextChanged.connect(lambda label, r=row: self._on_copper_type_changed(r, label))
            self.setCellWidget(row, _COL_DESC, combo)
        else:
            # Col 0 — clickable type toggle button ("Core" / "Prepreg" / "Mask")
            layer_type = _get_layer_type(data["material"])
            type_btn = QPushButton(layer_type)
            type_btn.setStyleSheet(
                f"QPushButton {{ background-color: {color.name()}; color: #111; "
                f"border: none; font-weight: bold; }}"
                f"QPushButton:hover {{ background-color: {color.darker(108).name()}; }}"
            )
            type_btn.clicked.connect(lambda _, r=row: self._toggle_type(r))
            self.setCellWidget(row, _COL_LAYER, type_btn)

            # Col 1 — glass construction combo ("1080 — 76%", "7628 — 42%", …)
            combo = QComboBox()
            combo.setStyleSheet(f"background-color: {color.name()}; color: #111;")
            combo.addItems(_GLASS_CONSTRUCTIONS)
            saved = data.get("construction", "")
            if saved in _GLASS_CONSTRUCTIONS:
                combo.setCurrentText(saved)
            combo.setEnabled(layer_type != "Mask")
            combo.currentTextChanged.connect(lambda _, r=row: self.stackup_changed.emit())
            self.setCellWidget(row, _COL_DESC, combo)

        # Col 2 — Thickness (mils)
        spinbox = QDoubleSpinBox()
        spinbox.setRange(0.0, 500.0)
        spinbox.setDecimals(2)
        spinbox.setSuffix(" mils")
        spinbox.setValue(data["finish_thickness_mil"])
        spinbox.setStyleSheet(_spinbox_style(color))
        spinbox.valueChanged.connect(lambda _: self.stackup_changed.emit())
        self.setCellWidget(row, _COL_THICK, spinbox)

        # Col 3 — Gerber file picker (copper only)
        if is_copper:
            cell = _GerberCell()
            cell.setStyleSheet(f"background-color: {color.darker(110).name()};")
            if data.get("gerber_path"):
                cell.set_path(data["gerber_path"])
            self.setCellWidget(row, _COL_GERBER, cell)
        else:
            empty = QLabel()
            empty.setStyleSheet(f"background-color: {color.name()};")
            self.setCellWidget(row, _COL_GERBER, empty)

    def _on_type_changed(self, row: int, type_str: str):
        """Update colours and state for a dielectric row after its type changes."""
        full_mat = _compose_material(self._active_material_set, type_str)
        color = QColor(_material_hex(full_mat))

        btn = self.cellWidget(row, _COL_LAYER)
        if isinstance(btn, QPushButton):
            btn.setText(type_str)
            btn.setStyleSheet(
                f"QPushButton {{ background-color: {color.name()}; color: #111; "
                f"border: none; font-weight: bold; }}"
                f"QPushButton:hover {{ background-color: {color.darker(108).name()}; }}"
            )
        combo = self.cellWidget(row, _COL_DESC)
        if isinstance(combo, QComboBox):
            combo.setEnabled(type_str != "Mask")
            combo.setStyleSheet(f"background-color: {color.name()}; color: #111;")
        spinbox = self.cellWidget(row, _COL_THICK)
        if isinstance(spinbox, QDoubleSpinBox):
            spinbox.setStyleSheet(_spinbox_style(color))
            thick = _DEFAULT_THICKNESS.get(full_mat)
            if thick is not None:
                spinbox.blockSignals(True)
                spinbox.setValue(thick)
                spinbox.blockSignals(False)
        gerber = self.cellWidget(row, _COL_GERBER)
        if isinstance(gerber, QLabel):
            gerber.setStyleSheet(f"background-color: {color.name()};")
        self._refresh_copper_options()
        self.stackup_changed.emit()

    def _toggle_type(self, row: int) -> None:
        """Clicking the layer-type button cycles Core ↔ Prepreg (↔ Mask for external rows)."""
        btn = self.cellWidget(row, _COL_LAYER)
        if not isinstance(btn, QPushButton):
            return
        current = btn.text()
        last = self.rowCount() - 1
        is_external = (row == 0 or row == last)
        cycle = ["Mask", "Core", "Prepreg"] if is_external else ["Core", "Prepreg"]
        idx = cycle.index(current) if current in cycle else 0
        self._on_type_changed(row, cycle[(idx + 1) % len(cycle)])

    def _on_copper_type_changed(self, row: int, label: str):
        thickness = _ALL_COPPER_TYPES.get(label)
        if thickness is not None:
            spinbox = self.cellWidget(row, _COL_THICK)
            if isinstance(spinbox, QDoubleSpinBox):
                spinbox.blockSignals(True)
                spinbox.setValue(thickness)
                spinbox.blockSignals(False)
        self.stackup_changed.emit()

    def _adjacent_dielectric_material(self, row: int, direction: int) -> str | None:
        """Walk up (direction=-1) or down (+1) and return the composed material name of the first dielectric found."""
        i = row + direction
        while 0 <= i < self.rowCount():
            if not self._is_copper_row(i):
                btn = self.cellWidget(i, _COL_LAYER)
                if isinstance(btn, QPushButton):
                    return _compose_material(self._active_material_set, btn.text())
            i += direction
        return None

    def _copper_options_for_row(self, row: int) -> dict[str, float]:
        above = self._adjacent_dielectric_material(row, -1)
        below = self._adjacent_dielectric_material(row, +1)
        # Foil+plating only when prepreg is on both sides (or external with no core neighbor)
        if _is_prepreg(above) and _is_prepreg(below):
            return _FOIL_PLATING_TYPES
        if _is_core(above) or _is_core(below):
            return _OZ_TYPES
        return _FOIL_PLATING_TYPES

    def _renumber(self):
        copper_n = 0
        for i in range(self.rowCount()):
            if self._is_copper_row(i):
                copper_n += 1
                lbl = self.cellWidget(i, _COL_LAYER)
                if lbl:
                    lbl.setText(f"L{copper_n}")
            # Dielectric rows: type button text is managed by _on_type_changed
        self._refresh_material_options()
        self._refresh_copper_options()

    def _refresh_material_options(self):
        """If an internal row has Mask selected (e.g. after a move), reset it to Core."""
        last = self.rowCount() - 1
        for i in range(self.rowCount()):
            if self._is_copper_row(i):
                continue
            btn = self.cellWidget(i, _COL_LAYER)
            if not isinstance(btn, QPushButton):
                continue
            is_external = (i == 0 or i == last)
            if not is_external and btn.text() == "Mask":
                self._on_type_changed(i, "Core")

    def _refresh_copper_options(self):
        """Update each copper row's dropdown based on its adjacent dielectric materials."""
        for i in range(self.rowCount()):
            if not self._is_copper_row(i):
                continue
            combo = self.cellWidget(i, _COL_DESC)
            if not isinstance(combo, QComboBox):
                continue

            options = self._copper_options_for_row(i)
            current = combo.currentText()

            combo.blockSignals(True)
            combo.clear()
            combo.addItems(list(options.keys()))
            if current in options:
                combo.setCurrentText(current)
            else:
                combo.setCurrentIndex(0)
                # Auto-fill thickness for the new default type
                spinbox = self.cellWidget(i, _COL_THICK)
                if isinstance(spinbox, QDoubleSpinBox):
                    spinbox.blockSignals(True)
                    spinbox.setValue(list(options.values())[0])
                    spinbox.blockSignals(False)
            combo.blockSignals(False)

    def _swap(self, a: int, b: int):
        data_a = asdict(self._extract(a))
        data_b = asdict(self._extract(b))
        self._populate(a, data_b)
        self._populate(b, data_a)
        self._renumber()

    def _extract(self, row: int) -> StackupRow:
        is_copper = self._is_copper_row(row)

        spinbox = self.cellWidget(row, _COL_THICK)
        thickness = spinbox.value() if isinstance(spinbox, QDoubleSpinBox) else 0.0

        desc = self.cellWidget(row, _COL_DESC)

        if is_copper:
            material = desc.currentText() if isinstance(desc, QComboBox) else ""
            lbl = self.cellWidget(row, _COL_LAYER)
            text = lbl.text() if lbl else ""
            layer_num = int(text[1:]) if text.startswith("L") else None
            gerber_cell = self.cellWidget(row, _COL_GERBER)
            gerber_path = gerber_cell.gerber_path if isinstance(gerber_cell, _GerberCell) else None
            return StackupRow(row_type="copper", layer_number=layer_num,
                              gerber_path=gerber_path, material=material,
                              finish_thickness_mil=thickness)
        else:
            btn = self.cellWidget(row, _COL_LAYER)
            type_str = btn.text() if isinstance(btn, QPushButton) else "Core"
            material = _compose_material(self._active_material_set, type_str)
            construction = desc.currentText() if isinstance(desc, QComboBox) and type_str != "Mask" else ""
            return StackupRow(row_type="dielectric", material=material,
                              construction=construction,
                              finish_thickness_mil=thickness)
