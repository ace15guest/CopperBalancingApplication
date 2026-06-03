from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QHeaderView, QTableWidget, QTableWidgetItem

from src.models import ComparisonMetrics, GradientMetrics

_METRICS: list[tuple[str, str, str, str]] = [
    ("RMS Error",          "rms_error",            ".4f", "mm"),
    ("R²",                 "r_squared",             ".4f", ""),
    ("Pearson r",          "pearson",               ".4f", ""),
    ("Gradient Corr.",     "gradient_correlation",  ".4f", ""),
    ("Hotspot Overlap",    "hotspot_overlap",       ".3f", ""),
    ("IPC Bow Ratio",      "ipc_bow_ratio",         ".4f", "mm/mm"),
]

_GRAD_METRICS: list[tuple[str, str, str, str]] = [
    ("Angle Mean",         "angle_mean_deg",    ".2f", "°"),
    ("Angle Median",       "angle_median_deg",  ".2f", "°"),
    ("Angle p95",          "angle_p95_deg",     ".2f", "°"),
    ("Mag Ratio Mean",     "mag_ratio_mean",    ".3f", ""),
    ("Mag Ratio Median",   "mag_ratio_median",  ".3f", ""),
    ("Mag Ratio p05",      "mag_ratio_p05",     ".3f", ""),
    ("Mag Ratio p95",      "mag_ratio_p95",     ".3f", ""),
]

_HEADER_BG = QColor("#2a3a2a")


class ComparisonTable(QTableWidget):
    """Displays CLT vs. high-fidelity vs. measured metrics side-by-side.

    Gradient rows are hidden until ``set_gradient_metrics`` is called.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._grad_row_start = len(_METRICS) + 1   # +1 for section header
        self._total_rows = len(_METRICS) + 1 + len(_GRAD_METRICS)
        self._setup()

    def _setup(self) -> None:
        self.setColumnCount(3)
        self.setHorizontalHeaderLabels(["Metric", "CLT", "Hi-Fi"])
        self.setRowCount(self._total_rows)

        for row, (label, _, _, unit) in enumerate(_METRICS):
            display = f"{label}  ({unit})" if unit else label
            self._label_item(row, display)

        # Section header row for gradient block
        hdr_row = len(_METRICS)
        for col in range(3):
            item = QTableWidgetItem("Gradient Analysis" if col == 0 else "")
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            item.setBackground(_HEADER_BG)
            item.setForeground(QColor("#aaddaa"))
            self.setItem(hdr_row, col, item)

        for i, (label, _, _, unit) in enumerate(_GRAD_METRICS):
            display = f"{label}  ({unit})" if unit else label
            self._label_item(self._grad_row_start + i, display)

        hdr = self.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.verticalHeader().setVisible(False)
        self._clear_values()

        # Hide gradient rows until data is available
        for r in range(len(_METRICS), self._total_rows):
            self.setRowHidden(r, True)

    def _label_item(self, row: int, text: str) -> None:
        item = QTableWidgetItem(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled)
        self.setItem(row, 0, item)

    def _clear_values(self) -> None:
        for row in range(self._total_rows):
            if row == len(_METRICS):
                continue
            for col in (1, 2):
                item = QTableWidgetItem("—")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.setItem(row, col, item)

    def populate(self, clt: ComparisonMetrics | None, hifi: ComparisonMetrics | None) -> None:
        for row, (_, attr, fmt, _) in enumerate(_METRICS):
            for col, result in ((1, clt), (2, hifi)):
                text = format(getattr(result, attr), fmt) if result is not None else "—"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.setItem(row, col, item)

    def set_gradient_metrics(self, clt: GradientMetrics | None, hifi: GradientMetrics | None) -> None:
        """Show the gradient section and fill it with data."""
        for r in range(len(_METRICS), self._total_rows):
            self.setRowHidden(r, False)
        for i, (_, attr, fmt, _) in enumerate(_GRAD_METRICS):
            row = self._grad_row_start + i
            for col, result in ((1, clt), (2, hifi)):
                text = format(getattr(result, attr), fmt) if result is not None else "—"
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.setItem(row, col, item)
