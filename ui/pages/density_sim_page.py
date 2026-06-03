from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QGroupBox, QPlainTextEdit, QFileDialog, QSplitter,
    QComboBox, QRadioButton, QButtonGroup,
)

from ui.components.heatmap_view import HeatmapView

_PROJECT_ROOT = Path(__file__).parents[2]
_NPZ_ROOT     = _PROJECT_ROOT / "assets" / "processed_npz"

_OZ_THICKNESS_M = 1.4 * 25.4e-6   # 1 oz copper = 1.4 mil


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class _DensityWorker(QObject):
    log      = pyqtSignal(str)
    finished = pyqtSignal(object, object)   # (sum_map, imbalance_map) as NDArrays
    error    = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self._p = params

    def run(self):
        try:
            self._do_run()
        except Exception:
            import traceback
            self.error.emit(traceback.format_exc())

    def _do_run(self):
        from src.simulation.solver import load_stackup, compute_z_coordinates

        p            = self._p
        stackup_path = Path(p["stackup_path"])
        npz_folder   = Path(p["npz_folder"])
        blur_type    = p["blur_type"]
        sigma_px     = p["sigma_px"]
        kernel_px    = p["kernel_px"]

        # Parse stackup and compute z-coords (sets neutral axis)
        self.log.emit("Loading stackup...")
        layers   = load_stackup(stackup_path)
        z_coords = compute_z_coordinates(layers)
        active   = [l for l in layers if l.row_class != "ignore"]

        # Determine grid shape from first available NPZ
        grid_shape: tuple[int, int] = (200, 200)
        for layer in active:
            if layer.row_class == "copper_inner" and layer.gerber_path:
                npz_path = npz_folder / f"{layer.gerber_path.stem}.npz"
                if npz_path.exists():
                    grid_shape = np.load(npz_path)["density"].shape
                    break
        self.log.emit(f"Grid: {grid_shape[1]} x {grid_shape[0]} px")

        # Accumulate weighted density maps
        sum_map = np.zeros(grid_shape, dtype=np.float64)
        top_map = np.zeros(grid_shape, dtype=np.float64)
        bot_map = np.zeros(grid_shape, dtype=np.float64)

        copper_pairs = [
            (layer, z)
            for layer, z in zip(active, z_coords)
            if layer.row_class in ("copper_outer", "copper_inner")
        ]

        for layer, (z_bot, z_top) in copper_pairs:
            oz = layer.thickness_m / _OZ_THICKNESS_M
            z_mid = (z_bot + z_top) / 2.0

            if layer.row_class == "copper_outer":
                density = np.ones(grid_shape, dtype=np.float64)
            elif layer.gerber_path:
                npz_path = npz_folder / f"{layer.gerber_path.stem}.npz"
                if npz_path.exists():
                    raw = np.load(npz_path)["density"].astype(np.float64)
                    if raw.shape != grid_shape:
                        from PIL import Image
                        img = Image.fromarray(raw.astype(np.float32))
                        img = img.resize((grid_shape[1], grid_shape[0]), Image.BILINEAR)
                        raw = np.array(img, dtype=np.float64)
                    density = raw
                else:
                    self.log.emit(f"  L{layer.layer_number}: NPZ not found — using 0.5")
                    density = np.full(grid_shape, 0.5, dtype=np.float64)
            else:
                density = np.full(grid_shape, 0.5, dtype=np.float64)

            weighted = density * oz
            sum_map += weighted

            if z_mid >= 0.0:
                top_map += weighted
            else:
                bot_map += weighted

            side = "top" if z_mid >= 0.0 else "bot"
            self.log.emit(
                f"  L{str(layer.layer_number or '-'):<3} {layer.material:<22} "
                f"{oz:.2f} oz  z={z_mid*1e3:+.2f} mm  [{side}]"
            )

        imbalance_map = top_map - bot_map

        # Apply blur
        self.log.emit(f"\nApplying {blur_type} blur...")
        if blur_type == "Gaussian":
            from scipy.ndimage import gaussian_filter
            blurred_sum       = gaussian_filter(sum_map,       sigma=sigma_px)
            blurred_imbalance = gaussian_filter(imbalance_map, sigma=sigma_px)
        else:
            from scipy.ndimage import uniform_filter
            blurred_sum       = uniform_filter(sum_map,       size=kernel_px)
            blurred_imbalance = uniform_filter(imbalance_map, size=kernel_px)

        self.log.emit("Done.")
        self.finished.emit(blurred_sum, blurred_imbalance)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

class DensitySimPage(QWidget):

    density_result_ready = pyqtSignal(object, object, float, float)
    # Emits (sum_map: NDArray, imbalance_map: NDArray, board_w_mm: float, board_h_mm: float)

    def __init__(self):
        super().__init__()
        self._thread:     QThread | None      = None
        self._worker:     _DensityWorker | None = None
        self._board_w_mm: float               = 0.0
        self._board_h_mm: float               = 0.0
        self._sum_data:   np.ndarray | None   = None
        self._imb_data:   np.ndarray | None   = None
        self._build()
        _default_stackup = Path(
            r"C:\Users\Asa Guest\Documents\Projects\CopperBalancingApplication\Cu_Balancing_890K_StackUp.json"
        )
        if _default_stackup.exists():
            self._load_stackup(_default_stackup)
        self._board_name.setText("Cu_Bal_TV_Q1")

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---- Left: controls ----
        left        = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 4, 10)
        left_layout.setSpacing(8)

        left_layout.addWidget(self._build_board_group())
        left_layout.addWidget(self._build_blur_group())
        left_layout.addWidget(self._build_view_group())

        self._run_btn = QPushButton("Run Density Simulation")
        self._run_btn.setEnabled(False)
        self._run_btn.setFixedHeight(32)
        self._run_btn.clicked.connect(self._run)
        left_layout.addWidget(self._run_btn)

        left_layout.addWidget(self._build_log_group(), stretch=1)

        splitter.addWidget(left)

        # ---- Right: dual heatmaps ----
        right        = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 10, 10, 10)
        right_layout.setSpacing(4)

        self._heatmap_splitter = QSplitter(Qt.Orientation.Horizontal)

        self._heatmap_sum = HeatmapView()
        self._heatmap_imb = HeatmapView()
        self._heatmap_splitter.addWidget(self._heatmap_sum)
        self._heatmap_splitter.addWidget(self._heatmap_imb)
        self._heatmap_splitter.setStretchFactor(0, 1)
        self._heatmap_splitter.setStretchFactor(1, 1)
        self._heatmap_splitter.setHandleWidth(4)

        right_layout.addWidget(self._heatmap_splitter)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setHandleWidth(4)

        root.addWidget(splitter)

    def _build_board_group(self) -> QGroupBox:
        grp    = QGroupBox("Board")
        layout = QFormLayout(grp)
        layout.setSpacing(6)

        # Stackup picker
        row = QHBoxLayout()
        self._stackup_path = QLineEdit()
        self._stackup_path.setPlaceholderText("Select a stackup JSON file...")
        self._stackup_path.setReadOnly(True)
        row.addWidget(self._stackup_path)
        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_stackup)
        row.addWidget(browse_btn)
        layout.addRow("Stackup:", row)

        self._board_name = QLineEdit()
        self._board_name.setPlaceholderText("e.g. Cu_Bal_TV_Q1")
        self._board_name.textChanged.connect(self._update_npz_status)
        layout.addRow("Board name:", self._board_name)

        self._spin_dpi = QSpinBox()
        self._spin_dpi.setRange(10, 2000)
        self._spin_dpi.setValue(50)
        self._spin_dpi.setSuffix(" DPI")
        self._spin_dpi.valueChanged.connect(self._update_npz_status)
        layout.addRow("Resolution:", self._spin_dpi)

        self._lbl_npz = QLabel("NPZ: —")
        layout.addRow("", self._lbl_npz)

        return grp

    def _build_blur_group(self) -> QGroupBox:
        grp    = QGroupBox("Blur")
        layout = QFormLayout(grp)
        layout.setSpacing(6)

        # Blur type radio buttons
        type_row = QHBoxLayout()
        self._radio_gaussian = QRadioButton("Gaussian")
        self._radio_box      = QRadioButton("Box")
        self._radio_gaussian.setChecked(True)
        self._blur_group = QButtonGroup(self)
        self._blur_group.addButton(self._radio_gaussian)
        self._blur_group.addButton(self._radio_box)
        self._radio_gaussian.toggled.connect(self._on_blur_type_changed)
        type_row.addWidget(self._radio_gaussian)
        type_row.addWidget(self._radio_box)
        type_row.addStretch()
        layout.addRow("Type:", type_row)

        self._spin_sigma = QDoubleSpinBox()
        self._spin_sigma.setRange(0.1, 100.0)
        self._spin_sigma.setValue(5.0)
        self._spin_sigma.setSuffix(" mm")
        self._spin_sigma.setDecimals(1)
        self._lbl_sigma = QLabel("Sigma:")
        layout.addRow(self._lbl_sigma, self._spin_sigma)

        self._spin_kernel = QSpinBox()
        self._spin_kernel.setRange(1, 200)
        self._spin_kernel.setValue(10)
        self._spin_kernel.setSuffix(" mm")
        self._lbl_kernel = QLabel("Kernel size:")
        layout.addRow(self._lbl_kernel, self._spin_kernel)

        self._on_blur_type_changed()   # set initial visibility
        return grp

    def _build_view_group(self) -> QGroupBox:
        grp    = QGroupBox("View")
        layout = QFormLayout(grp)
        layout.setSpacing(6)

        self._view_combo = QComboBox()
        self._view_combo.addItems(["Side by side", "Sum only", "Imbalance only"])
        self._view_combo.currentIndexChanged.connect(self._apply_view_mode)
        layout.addRow("Display:", self._view_combo)

        return grp

    def _build_log_group(self) -> QGroupBox:
        grp    = QGroupBox("Log")
        layout = QVBoxLayout(grp)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(500)
        layout.addWidget(self._log)
        return grp

    # ------------------------------------------------------------------
    # Board / NPZ
    # ------------------------------------------------------------------

    def _browse_stackup(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Stackup", "", "Stackup JSON (*.json);;All Files (*)"
        )
        if path:
            self._load_stackup(Path(path))

    def _load_stackup(self, path: Path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            self._log_line(f"Error loading stackup: {e}")
            return

        self._stackup_path.setText(str(path))

        # Try to read board dims from Gerber outline
        for row in data.get("stackup", []):
            gp = row.get("gerber_path")
            if gp:
                gerber_folder = Path(gp).parent
                self._read_canvas_dims(gerber_folder)
                break

        self._update_npz_status()
        self._run_btn.setEnabled(bool(self._stackup_path.text()))

    def _read_canvas_dims(self, gerber_folder: Path):
        try:
            from src.ingestion.gerber_parser import _parse_gerber_extents
            outline = gerber_folder / "outline.gbr"
            if outline.exists():
                extents = _parse_gerber_extents(outline)
                if extents:
                    x_min, y_min, x_max, y_max = extents
                    self._board_w_mm = x_max - x_min
                    self._board_h_mm = y_max - y_min
        except Exception:
            pass

    def _update_npz_status(self):
        board_name  = self._board_name.text().strip()
        dpi         = self._spin_dpi.value()
        folder_name = f"{board_name}_{dpi}dpi" if board_name else None

        if folder_name and _NPZ_ROOT.exists():
            exact = _NPZ_ROOT / folder_name
            if exact.is_dir() and any(exact.glob("*.npz")):
                count = len(list(exact.glob("*.npz")))
                self._lbl_npz.setText(f"{folder_name} ({count} files) — ready")
                return

        self._lbl_npz.setText(
            f"{folder_name} — not found" if folder_name else "Enter board name"
        )

    def _resolve_npz_folder(self) -> Path | None:
        board_name = self._board_name.text().strip()
        dpi        = self._spin_dpi.value()
        if not board_name:
            return None
        folder = _NPZ_ROOT / f"{board_name}_{dpi}dpi"
        return folder if folder.is_dir() else None

    # ------------------------------------------------------------------
    # Blur controls
    # ------------------------------------------------------------------

    def _on_blur_type_changed(self):
        is_gaussian = self._radio_gaussian.isChecked()
        self._lbl_sigma.setVisible(is_gaussian)
        self._spin_sigma.setVisible(is_gaussian)
        self._lbl_kernel.setVisible(not is_gaussian)
        self._spin_kernel.setVisible(not is_gaussian)

    def _blur_params(self) -> dict:
        dpi = self._spin_dpi.value()
        px_per_mm = dpi / 25.4
        if self._radio_gaussian.isChecked():
            sigma_px = self._spin_sigma.value() * px_per_mm
            return {"blur_type": "Gaussian", "sigma_px": sigma_px, "kernel_px": 3}
        else:
            kernel_mm = self._spin_kernel.value()
            kernel_px = max(3, int(kernel_mm * px_per_mm))
            if kernel_px % 2 == 0:
                kernel_px += 1
            return {"blur_type": "Box", "sigma_px": 1.0, "kernel_px": kernel_px}

    # ------------------------------------------------------------------
    # View mode
    # ------------------------------------------------------------------

    def _apply_view_mode(self):
        mode = self._view_combo.currentText()
        if mode == "Side by side":
            self._heatmap_sum.setVisible(True)
            self._heatmap_imb.setVisible(True)
        elif mode == "Sum only":
            self._heatmap_sum.setVisible(True)
            self._heatmap_imb.setVisible(False)
        else:
            self._heatmap_sum.setVisible(False)
            self._heatmap_imb.setVisible(True)

        # Re-render if data is available
        if self._sum_data is not None:
            self._render_heatmaps()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _run(self):
        if self._thread and self._thread.isRunning():
            return

        stackup_path = self._stackup_path.text()
        npz_folder   = self._resolve_npz_folder()

        if not stackup_path:
            self._log_line("No stackup selected.")
            return
        if npz_folder is None:
            self._log_line("NPZ folder not found. Check board name and DPI.")
            return

        self._log.clear()
        self._sum_data = None
        self._imb_data = None
        self._run_btn.setEnabled(False)

        params = {
            "stackup_path": stackup_path,
            "npz_folder":   str(npz_folder),
            **self._blur_params(),
        }

        self._worker = _DensityWorker(params)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._log_line)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(lambda: self._run_btn.setEnabled(True))

        self._thread.start()

    def _log_line(self, msg: str):
        self._log.appendPlainText(msg)

    def _on_finished(self, sum_map: np.ndarray, imbalance_map: np.ndarray):
        self._sum_data = sum_map
        self._imb_data = imbalance_map
        self._render_heatmaps()
        self.density_result_ready.emit(sum_map, imbalance_map, self._board_w_mm, self._board_h_mm)

    def _on_error(self, msg: str):
        self._log_line(f"ERROR:\n{msg}")

    def _render_heatmaps(self):
        ny, nx = self._sum_data.shape
        x_mm = np.linspace(0, self._board_w_mm, nx) if self._board_w_mm else np.arange(nx)
        y_mm = np.linspace(0, self._board_h_mm, ny) if self._board_h_mm else np.arange(ny)

        if self._heatmap_sum.isVisible():
            self._heatmap_sum.show_heatmap(
                z=self._sum_data,
                x=x_mm,
                y=y_mm,
                title="Weighted Cu Density (oz)",
            )
        if self._heatmap_imb.isVisible():
            self._heatmap_imb.show_heatmap(
                z=self._imb_data,
                x=x_mm,
                y=y_mm,
                title="Cu Imbalance — Top vs Bottom (oz)",
            )
