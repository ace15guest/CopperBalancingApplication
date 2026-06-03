from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.analysis.alignment import align
from src.analysis.gradient import gradient_analysis
from src.analysis.metrics import compute_metrics, difference_map
from src.ingestion.akrometrix import load_akrometrix_dat
from src.models import MeasurementData, SimResult
from src.processing.denoise import denoise
from src.processing.inpainting import fill_missing
from ui.components.comparison_table import ComparisonTable
from ui.components.heatmap_view import HeatmapView


class ComparePage(QWidget):
    """Side-by-side comparison of simulation outputs vs. Akrometrix measurements."""

    def __init__(self):
        super().__init__()
        self._measurement: MeasurementData | None = None
        self._aligned_measurement: MeasurementData | None = None
        self._sim_clt: SimResult | None = None
        self._sim_hifi: SimResult | None = None
        self._view_mode: str = "heatmap"  # "heatmap" | "surface"
        self._plot_cache: dict[str, tuple] = {}
        # Density simulation result (from DensitySimPage)
        self._density_sum: np.ndarray | None = None
        self._density_imb: np.ndarray | None = None
        self._density_x:   np.ndarray | None = None
        self._density_y:   np.ndarray | None = None
        self._build()
        self._dat_path.setText(self._DEFAULT_DAT_DIR)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # --- Top bar: .dat file picker ---
        top = QHBoxLayout()
        self._dat_path = QLineEdit()
        self._dat_path.setPlaceholderText("Akrometrix .dat file")
        self._dat_path.setReadOnly(True)
        top.addWidget(self._dat_path)

        browse_btn = QPushButton("Browse .dat")
        browse_btn.setFixedWidth(110)
        browse_btn.clicked.connect(self._browse)
        top.addWidget(browse_btn)

        load_btn = QPushButton("Load")
        load_btn.setFixedWidth(70)
        load_btn.clicked.connect(self._load)
        top.addWidget(load_btn)

        top.addWidget(QLabel("±"))
        self._rot_limit_spin = QDoubleSpinBox()
        self._rot_limit_spin.setRange(0.5, 45.0)
        self._rot_limit_spin.setSingleStep(1.0)
        self._rot_limit_spin.setValue(10.0)
        self._rot_limit_spin.setSuffix("°")
        self._rot_limit_spin.setFixedWidth(68)
        self._rot_limit_spin.setToolTip("Maximum rotation search range (±degrees)")
        top.addWidget(self._rot_limit_spin)

        self._align_btn = QPushButton("Align")
        self._align_btn.setFixedWidth(70)
        self._align_btn.setEnabled(False)
        self._align_btn.setToolTip(
            "Apply a small rotation correction (limited to ± the angle shown)\n"
            "then resample the measurement to the simulation grid resolution.\n"
            "Run before comparing."
        )
        self._align_btn.clicked.connect(self._run_alignment)
        top.addWidget(self._align_btn)

        self._align_status = QLabel("")
        self._align_status.setStyleSheet("color: #888; font-style: italic;")
        top.addWidget(self._align_status)

        self._grad_btn = QPushButton("Gradient Analysis")
        self._grad_btn.setFixedWidth(140)
        self._grad_btn.setEnabled(False)
        self._grad_btn.setToolTip(
            "Compute per-pixel gradient angle difference and magnitude ratio\n"
            "between simulation and measurement."
        )
        self._grad_btn.clicked.connect(self._run_gradient_analysis)
        top.addWidget(self._grad_btn)

        self._fill_cb = QCheckBox("Fill missing (biharmonic)")
        self._fill_cb.setChecked(False)
        self._fill_cb.setToolTip(
            "Predict missing pixels by solving the thin-plate bending equation (∇⁴u = 0).\n"
            "Physically appropriate for PCB warpage; adds a few seconds per file."
        )
        top.addWidget(self._fill_cb)

        self._denoise_cb = QCheckBox("Denoise")
        self._denoise_cb.setChecked(False)
        self._denoise_cb.setToolTip(
            "Three-step: large median pre-filter → MAD outlier clip → Gaussian smooth.\n"
            "Applied on Load; re-load to change settings."
        )
        top.addWidget(self._denoise_cb)

        top.addWidget(QLabel("Med:"))
        self._median_spin = QSpinBox()
        self._median_spin.setRange(0, 201)
        self._median_spin.setValue(0)
        self._median_spin.setSingleStep(2)
        self._median_spin.setSuffix(" px")
        self._median_spin.setFixedWidth(72)
        self._median_spin.setToolTip(
            "Large-median pre-filter window (pixels). Use odd values.\n"
            "Removes soldermask/copper step-function offsets that Gaussian\n"
            "cannot eliminate. Set larger than your biggest pad. 0 = off."
        )
        top.addWidget(self._median_spin)

        top.addWidget(QLabel("σ:"))
        self._sigma_spin = QDoubleSpinBox()
        self._sigma_spin.setRange(0.0, 100.0)
        self._sigma_spin.setValue(2.0)
        self._sigma_spin.setSingleStep(1.0)
        self._sigma_spin.setDecimals(1)
        self._sigma_spin.setSuffix(" px")
        self._sigma_spin.setFixedWidth(72)
        self._sigma_spin.setToolTip("Gaussian smoothing radius (pixels). Applied after median pre-filter.")
        top.addWidget(self._sigma_spin)

        self._view_toggle = QPushButton("Switch to 3D")
        self._view_toggle.setFixedWidth(110)
        self._view_toggle.setToolTip(
            "Toggle between flat heatmap and interactive 3D surface.\n"
            "In 3D mode, looking straight down reproduces the heatmap view;\n"
            "tilt with the mouse to inspect warpage shape."
        )
        self._view_toggle.clicked.connect(self._toggle_view_mode)
        top.addWidget(self._view_toggle)

        top.addWidget(QLabel("Center:"))
        self._crop_spin = QDoubleSpinBox()
        self._crop_spin.setRange(10.0, 100.0)
        self._crop_spin.setValue(100.0)
        self._crop_spin.setSingleStep(5.0)
        self._crop_spin.setDecimals(0)
        self._crop_spin.setSuffix("%")
        self._crop_spin.setFixedWidth(72)
        self._crop_spin.setToolTip(
            "Keep only the central N% of each panel.\n"
            "Reduces to 10% at minimum; 100% shows the full extent."
        )
        self._crop_spin.valueChanged.connect(self._on_crop_changed)
        top.addWidget(self._crop_spin)

        layout.addLayout(top)

        self._sub_tabs = QTabWidget()
        layout.addWidget(self._sub_tabs, stretch=1)

        # ---- Tab 1: CLT vs Measured ----
        clt_tab = QWidget()
        clt_layout = QVBoxLayout(clt_tab)
        clt_layout.setContentsMargins(0, 4, 0, 0)
        clt_layout.setSpacing(4)

        heatmap_row = QSplitter(Qt.Orientation.Horizontal)
        self._sim_heatmap, sim_wrap = self._labeled_panel("Simulation")
        heatmap_row.addWidget(sim_wrap)
        self._meas_heatmap, meas_wrap = self._labeled_panel("Measured (Akrometrix)")
        heatmap_row.addWidget(meas_wrap)
        self._diff_heatmap, diff_wrap = self._labeled_panel("Difference  (Sim − Measured)")
        heatmap_row.addWidget(diff_wrap)
        for i in range(3):
            heatmap_row.setStretchFactor(i, 1)
        clt_layout.addWidget(heatmap_row, stretch=1)

        self._grad_row = QSplitter(Qt.Orientation.Horizontal)
        self._angle_heatmap, angle_wrap = self._labeled_panel("Gradient Angle Diff (°)")
        self._mag_heatmap, mag_wrap = self._labeled_panel("Gradient Mag Ratio")
        self._grad_row.addWidget(angle_wrap)
        self._grad_row.addWidget(mag_wrap)
        self._grad_row.setStretchFactor(0, 1)
        self._grad_row.setStretchFactor(1, 1)
        self._grad_row.setVisible(False)
        clt_layout.addWidget(self._grad_row, stretch=1)

        self._table = ComparisonTable()
        self._table.setMaximumHeight(260)
        clt_layout.addWidget(self._table)

        self._sub_tabs.addTab(clt_tab, "CLT vs Measured")

        # ---- Tab 2: Density vs Measured ----
        self._sub_tabs.addTab(self._build_density_tab(), "Density vs Measured")

    def _build_density_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)

        row = QSplitter(Qt.Orientation.Horizontal)
        self._density_sum_heatmap, sum_wrap = self._labeled_panel("Weighted Cu Density (oz·sum)")
        self._density_imb_heatmap, imb_wrap = self._labeled_panel("Cu Imbalance Top−Bot (oz)")
        self._density_meas_heatmap, meas_wrap = self._labeled_panel("Measured Warpage (Akrometrix)")
        row.addWidget(sum_wrap)
        row.addWidget(imb_wrap)
        row.addWidget(meas_wrap)
        for i in range(3):
            row.setStretchFactor(i, 1)
        layout.addWidget(row, stretch=1)

        self._density_corr_lbl = QLabel("Run Density Simulation then load a .dat file to see correlation.")
        self._density_corr_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._density_corr_lbl.setStyleSheet("color: #888; font-style: italic; padding: 4px;")
        layout.addWidget(self._density_corr_lbl)

        return tab

    def _labeled_panel(self, title: str) -> tuple[HeatmapView, QWidget]:
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(2)
        lbl = QLabel(title)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        v.addWidget(lbl)
        heatmap = HeatmapView()
        v.addWidget(heatmap)
        return heatmap, container

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    _DEFAULT_DAT_DIR = (
        r"C:\Users\Asa Guest\Documents\Projects\CopperBalancingApplication\assets\AkroFiles\TopDatFiles\ACCL-890K\Q1\ACCL-890K-01-Q1_Top_Global.dat"
    )

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Akrometrix .dat File", self._DEFAULT_DAT_DIR, "DAT Files (*.dat);;All Files (*)"
        )
        if path:
            self._dat_path.setText(path)

    def _load(self) -> None:
        path_str = self._dat_path.text().strip()
        if not path_str:
            QMessageBox.warning(self, "No File", "Select a .dat file first.")
            return
        try:
            measurement = load_akrometrix_dat(Path(path_str))
            if self._fill_cb.isChecked():
                measurement = fill_missing(measurement)
            if self._denoise_cb.isChecked():
                measurement = denoise(
                    measurement,
                    sigma=self._sigma_spin.value(),
                    median_size=self._median_spin.value(),
                )
            self._measurement = measurement
            self._aligned_measurement = None
            self._align_status.setText("Not aligned")
            self._align_status.setStyleSheet("color: #cc8844; font-style: italic;")
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", str(exc))
            return
        self._refresh()

    # ------------------------------------------------------------------
    # Public: called by the simulate page when results arrive
    # ------------------------------------------------------------------

    def load_sim_results(
        self,
        clt: SimResult | None = None,
        hifi: SimResult | None = None,
    ) -> None:
        self._sim_clt = clt
        self._sim_hifi = hifi
        self._refresh()

    def load_density_result(
        self,
        sum_map: np.ndarray,
        imbalance_map: np.ndarray,
        board_w_mm: float,
        board_h_mm: float,
    ) -> None:
        ny, nx = sum_map.shape
        self._density_sum = sum_map
        self._density_imb = imbalance_map
        self._density_x   = np.linspace(0, board_w_mm, nx) if board_w_mm else np.arange(nx, dtype=float)
        self._density_y   = np.linspace(0, board_h_mm, ny) if board_h_mm else np.arange(ny, dtype=float)
        self._refresh_density()

    # ------------------------------------------------------------------
    # Display logic
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        self._refresh_measurement()
        self._refresh_sim()
        self._refresh_diff_and_metrics()

    @staticmethod
    def _center_crop(z, x, y, pct: float):
        if pct >= 100.0:
            return z, x, y
        rows, cols = z.shape
        r_margin = max(0, int(rows * (1.0 - pct / 100.0) / 2))
        c_margin = max(0, int(cols * (1.0 - pct / 100.0) / 2))
        r0, r1 = r_margin, rows - r_margin
        c0, c1 = c_margin, cols - c_margin
        if r1 <= r0 or c1 <= c0:
            return z, x, y
        return z[r0:r1, c0:c1], x[c0:c1], y[r0:r1]

    def _show(self, view: "HeatmapView", key: str, z, x, y, title: str) -> None:
        self._plot_cache[key] = (z, x, y, title)
        pct = self._crop_spin.value()
        zc, xc, yc = self._center_crop(z, x, y, pct)
        if self._view_mode == "surface":
            view.show_surface(zc, xc, yc, title)
        else:
            view.show_heatmap(zc, xc, yc, title)

    def _all_panel_views(self) -> dict[str, "HeatmapView"]:
        return {
            "sim":         self._sim_heatmap,
            "meas":        self._meas_heatmap,
            "diff":        self._diff_heatmap,
            "angle":       self._angle_heatmap,
            "mag":         self._mag_heatmap,
            "dens_sum":    self._density_sum_heatmap,
            "dens_imb":    self._density_imb_heatmap,
            "dens_meas":   self._density_meas_heatmap,
        }

    def _on_crop_changed(self) -> None:
        pct = self._crop_spin.value()
        for key, view in self._all_panel_views().items():
            if key in self._plot_cache:
                z, x, y, title = self._plot_cache[key]
                zc, xc, yc = self._center_crop(z, x, y, pct)
                if self._view_mode == "surface":
                    view.show_surface(zc, xc, yc, title)
                else:
                    view.show_heatmap(zc, xc, yc, title)

    def _toggle_view_mode(self) -> None:
        self._view_mode = "surface" if self._view_mode == "heatmap" else "heatmap"
        self._view_toggle.setText(
            "Switch to 2D" if self._view_mode == "surface" else "Switch to 3D"
        )
        pct = self._crop_spin.value()
        for key, view in self._all_panel_views().items():
            if key in self._plot_cache:
                z, x, y, title = self._plot_cache[key]
                zc, xc, yc = self._center_crop(z, x, y, pct)
                if self._view_mode == "surface":
                    view.show_surface(zc, xc, yc, title)
                else:
                    view.show_heatmap(zc, xc, yc, title)

    def _refresh_measurement(self) -> None:
        m = self._aligned_measurement or self._measurement
        if m is None:
            return
        label = "Measured — aligned (mm)" if self._aligned_measurement else "Measured — raw (mm)"
        z, x, y = _maybe_downsample(m.displacement, m.x_coords, m.y_coords, max_dim=600)
        self._show(self._meas_heatmap, "meas", z, x, y, label)
        # Mirror to density tab
        self._show(self._density_meas_heatmap, "dens_meas", z, x, y, label)
        self._update_density_correlation()

    def _refresh_density(self) -> None:
        if self._density_sum is None:
            return
        zs, xs, ys = _maybe_downsample(self._density_sum, self._density_x, self._density_y, max_dim=600)
        self._show(self._density_sum_heatmap, "dens_sum", zs, xs, ys, "Weighted Cu Density (oz·sum)")
        zi, xi, yi = _maybe_downsample(self._density_imb, self._density_x, self._density_y, max_dim=600)
        self._show(self._density_imb_heatmap, "dens_imb", zi, xi, yi, "Cu Imbalance Top−Bot (oz)")
        self._update_density_correlation()

    def _update_density_correlation(self) -> None:
        if self._density_imb is None:
            return
        m = self._aligned_measurement or self._measurement
        if m is None:
            return
        try:
            from scipy.ndimage import zoom
            imb = self._density_imb.astype(np.float64)
            meas = m.displacement.astype(np.float64)
            zy = meas.shape[0] / imb.shape[0]
            zx = meas.shape[1] / imb.shape[1]
            imb_rs = zoom(imb, (zy, zx), order=1)
            valid = np.isfinite(meas)
            d = meas[valid].ravel()
            i = imb_rs[valid].ravel()
            if len(d) < 4:
                return
            r = float(np.corrcoef(d, i)[0, 1])
            self._density_corr_lbl.setText(
                f"Pearson R (imbalance vs warpage): {r:+.4f}"
                + ("  — strong positive correlation" if r > 0.7 else
                   "  — strong negative correlation" if r < -0.7 else "")
            )
            self._density_corr_lbl.setStyleSheet(
                f"color: {'#88cc88' if abs(r) > 0.7 else '#ccaa44'}; padding: 4px;"
            )
        except Exception:
            pass

    def _refresh_sim(self) -> None:
        # Prefer hi-fi; fall back to CLT
        sim = self._sim_hifi or self._sim_clt
        if sim is None:
            return
        z, x, y = _maybe_downsample(sim.displacement, sim.x_coords, sim.y_coords, max_dim=600)
        label = "Hi-Fi simulation (mm)" if self._sim_hifi else "CLT simulation (mm)"
        self._show(self._sim_heatmap, "sim", z, x, y, label)

    def _refresh_diff_and_metrics(self) -> None:
        sim = self._sim_hifi or self._sim_clt
        can_compare = sim is not None and self._measurement is not None
        self._align_btn.setEnabled(can_compare)
        self._grad_btn.setEnabled(can_compare and self._aligned_measurement is not None)

        # Use aligned measurement for metrics if available; raw otherwise.
        # Raw comparison will be wrong (coordinate mismatch) but still shown
        # so the user can see something before alignment.
        meas = self._aligned_measurement if self._aligned_measurement is not None else self._measurement

        if not can_compare or meas is None:
            self._table.populate(None, None)
            return

        try:
            diff = difference_map(sim, meas)
            z, x, y = _maybe_downsample(diff, sim.x_coords, sim.y_coords, max_dim=600)
            self._show(self._diff_heatmap, "diff", z, x, y, "Difference (mm)")
        except Exception:
            pass

        try:
            clt_m = compute_metrics(self._sim_clt, meas) if self._sim_clt else None
            hifi_m = compute_metrics(self._sim_hifi, meas) if self._sim_hifi else None
            self._table.populate(clt_m, hifi_m)
        except Exception as exc:
            QMessageBox.warning(self, "Metrics Error", str(exc))

    def _run_alignment(self) -> None:
        sim = self._sim_hifi or self._sim_clt
        if sim is None or self._measurement is None:
            return
        self._align_btn.setEnabled(False)
        self._align_status.setText("Aligning…")
        try:
            aligned, params = align(
                sim.displacement, self._measurement,
                rotation_limit_deg=self._rot_limit_spin.value(),
            )
            self._aligned_measurement = aligned
            angle = params.get("angle_deg", 0.0)
            sx = params.get("scale_x", 1.0)
            sy = params.get("scale_y", 1.0)
            self._align_status.setText(
                f"Aligned  rot={angle:.2f}°  resample=({sx:.3f}×, {sy:.3f}×)"
            )
            self._align_status.setStyleSheet("color: #88cc88;")
        except Exception as exc:
            self._align_status.setText(f"Align failed: {exc}")
            self._align_status.setStyleSheet("color: #cc6666;")
        finally:
            self._align_btn.setEnabled(True)
        self._refresh()

    def _run_gradient_analysis(self) -> None:
        sim = self._sim_hifi or self._sim_clt
        meas = self._aligned_measurement or self._measurement
        if sim is None or meas is None:
            return
        try:
            clt_g, angle_diff, mag_ratio = gradient_analysis(sim, meas)
            hifi_g = None
            if self._sim_hifi and self._sim_clt:
                hifi_g, _, _ = gradient_analysis(self._sim_hifi, meas)
                clt_g, _, _ = gradient_analysis(self._sim_clt, meas)

            self._table.set_gradient_metrics(clt_g, hifi_g)

            az, ax, ay = _maybe_downsample(angle_diff, sim.x_coords, sim.y_coords, max_dim=600)
            mz, mx, my = _maybe_downsample(mag_ratio,  sim.x_coords, sim.y_coords, max_dim=600)
            self._show(self._angle_heatmap, "angle", az, ax, ay, "Angle diff (°)")
            self._show(self._mag_heatmap,   "mag",   mz, mx, my, "Mag ratio")
            self._grad_row.setVisible(True)
        except Exception as exc:
            QMessageBox.warning(self, "Gradient Analysis Error", str(exc))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _maybe_downsample(
    z: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    max_dim: int = 600,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce grid resolution for display if either dimension exceeds max_dim."""
    rows, cols = z.shape
    row_step = max(1, rows // max_dim)
    col_step = max(1, cols // max_dim)
    if row_step == 1 and col_step == 1:
        return z, x, y
    return z[::row_step, ::col_step], x[::col_step], y[::row_step]
