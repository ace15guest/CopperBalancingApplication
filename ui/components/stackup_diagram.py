from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QFont
from PyQt6.QtCore import Qt
from src.models import StackupRow

COPPER_COLOR = "#D4820A"
MATERIAL_COLORS: dict[str, str] = {
    "Liquid PhotoImageable Mask": "#1B5E20",
    "PrePreg 1080":               "#C8D890",
    "PrePreg 1651":               "#BDD082",
    "PrePreg 3080":               "#B2C874",
    "FR4 Core":                   "#8CA85A",
    "Copper Foil":                COPPER_COLOR,
}
_FALLBACK_COLOR = "#888888"
_MIN_ROW_PX = 10      # minimum height per layer so thin layers stay visible
_MARGIN = 12


def material_color(row: StackupRow) -> QColor:
    if row.row_type == "copper":
        return QColor(COPPER_COLOR)
    c = MATERIAL_COLORS.get(row.material)
    if c:
        return QColor(c)
    if row.material.endswith(" Core"):
        return QColor(MATERIAL_COLORS["FR4 Core"])
    if row.material.endswith(" Prepreg"):
        return QColor(MATERIAL_COLORS["PrePreg 1080"])
    return QColor(_FALLBACK_COLOR)


class StackupDiagram(QWidget):
    """
    Painted cross-section diagram showing all stackup rows proportionally.
    Call update_rows() whenever the stackup changes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[StackupRow] = []
        self.setMinimumWidth(160)

    def update_rows(self, rows: list[StackupRow]) -> None:
        self._rows = rows
        self.update()

    def paintEvent(self, event):
        if not self._rows:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        total_thickness = sum(r.finish_thickness_mil for r in self._rows) or 1.0
        available_h = self.height() - 2 * _MARGIN
        draw_w = self.width() - 2 * _MARGIN

        # Reserve minimum pixels for each row, distribute remainder proportionally
        min_total = _MIN_ROW_PX * len(self._rows)
        extra = max(0, available_h - min_total)

        font = QFont("Segoe UI", 7)
        painter.setFont(font)

        y = _MARGIN
        copper_count = 0
        for row in self._rows:
            proportion = row.finish_thickness_mil / total_thickness
            row_h = _MIN_ROW_PX + int(proportion * extra)

            color = material_color(row)
            painter.fillRect(_MARGIN, y, draw_w, row_h, color)
            painter.setPen(QColor("#333333"))
            painter.drawRect(_MARGIN, y, draw_w, row_h)

            if row.row_type == "copper":
                copper_count += 1
                label = f"L{row.layer_number or copper_count}"
                painter.setPen(QColor("#111111"))
                painter.drawText(
                    _MARGIN + 2, y, draw_w - 4, row_h,
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                    label,
                )
            elif row_h >= 14:
                short = row.material.replace("Liquid PhotoImageable Mask", "LPI Mask")
                painter.setPen(QColor("#111111"))
                painter.drawText(
                    _MARGIN + 2, y, draw_w - 4, row_h,
                    Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                    short,
                )

            y += row_h

        painter.end()
