from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QGroupBox, QPlainTextEdit, QFileDialog, QSplitter,
)

from src.models import SimResult
from ui.components.heatmap_view import HeatmapView

_PROJECT_ROOT = Path(__file__).parents[2]
_LIBRARY_PATH = _PROJECT_ROOT / "data" / "materials.json"
_NPZ_ROOT     = _PROJECT_ROOT / "assets" / "processed_npz"
_PNG_ROOT     = _PROJECT_ROOT / "assets" / "processed_pngs"


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _SimWorker(QObject):
    log      = pyqtSignal(str)
    finished = pyqtSignal(object)   # SimulationResult
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
        from src.ingestion.gerber_parser import (
            _parse_gerber_extents,
            generate_outline_from_extrema,
        )
        from src.simulation.solver import run_simulation

        p = self._p
        stackup_path  = Path(p["stackup_path"])
        gerber_folder = Path(p["gerber_folder"])
        dpi           = p["dpi"]

        # Board canvas dimensions from outline
        self.log.emit("Reading board outline...")
        outline = gerber_folder / "outline.gbr"
        if not outline.exists():
            self.log.emit("No outline.gbr — generating from Gerber extents...")
            generate_outline_from_extrema(gerber_folder, outline)
        extents = _parse_gerber_extents(outline)
        if extents is None:
            raise ValueError("Could not determine board dimensions from outline.gbr")
        x_min, y_min, x_max, y_max = extents
        board_w_mm = x_max - x_min
        board_h_mm = y_max - y_min
        self.log.emit(f"Board canvas: {board_w_mm:.1f} mm x {board_h_mm:.1f} mm")

        # Resolve NPZ folder (find → generate from PNGs → render from Gerbers)
        npz_folder = self._resolve_npz(gerber_folder, p["board_name"], dpi)

        # Run simulation
        self.log.emit("Running CLT simulation...")
        result = run_simulation(
            json_path=stackup_path,
            library_path=_LIBRARY_PATH,
            process_config={
                "T_press_c":         p["T_peak"],
                "T_ambient_c":       p["T_start"],
                "T_vitrification_c": p["T_vit"],
                "board_width_m":     board_w_mm / 1000.0,
                "board_height_m":    board_h_mm / 1000.0,
                "material_set":      p["material_set"],
            },
            npz_folder=npz_folder,
        )

        if result.b_matrix_warning:
            self.log.emit(
                "WARNING: B-matrix coupling is significant — result is a lower bound."
            )
        self.log.emit(
            f"Done.  Peak bow: {result.peak_bow_mm:.4f} mm  "
            f"Bow/span ratio: {result.bow_span_ratio:.6f}"
        )
        self.finished.emit(result)

    def _resolve_npz(self, gerber_folder: Path, board_name: str, dpi: int) -> Path | None:
        suffix      = f"_{dpi}dpi"
        folder_name = f"{board_name}{suffix}"

        # 1. Exact named NPZ folder
        exact_npz = _NPZ_ROOT / folder_name
        if exact_npz.is_dir() and any(exact_npz.glob("*.npz")):
            self.log.emit(f"Using NPZ folder: {folder_name}")
            return exact_npz

        # 2. Exact named PNG folder → generate NPZ
        exact_png = _PNG_ROOT / folder_name
        if exact_png.is_dir() and any(exact_png.glob("*.png")):
            self.log.emit(f"Generating NPZ from PNGs in {folder_name}...")
            from src.ingestion.gerber_parser import BatchResult
            from src.processing.rasterizer import batch_png_to_npz
            png_paths = sorted(exact_png.glob("*.png"))
            batch = BatchResult(
                design_name=board_name,
                dpi=dpi,
                output_folder=exact_png,
                outline_path=exact_png / "outline.png",
                png_paths=png_paths,
            )
            res = batch_png_to_npz(batch, output_root=_NPZ_ROOT)
            self.log.emit(f"Created {len(res.npz_paths)} NPZ files.")
            return res.output_folder

        # 3. Render Gerbers → PNGs → NPZ
        self.log.emit("Rendering Gerbers to PNG (this may take a while)...")
        from src.ingestion.gerber_parser import convert_folder
        from src.processing.rasterizer import batch_png_to_npz
        batch = convert_folder(
            gerber_folder=gerber_folder,
            output_root=_PNG_ROOT,
            design_name=board_name,
            dpi=dpi,
        )
        self.log.emit(f"Rendered {len(batch.png_paths)} PNGs.")
        res = batch_png_to_npz(batch, output_root=_NPZ_ROOT)
        self.log.emit(f"Created {len(res.npz_paths)} NPZ files.")
        return res.output_folder


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

class SimulatePage(QWidget):

    sim_result_ready = pyqtSignal(object)   # emits SimResult when CLT finishes

    def __init__(self):
        super().__init__()
        self._thread:        QThread | None   = None
        self._worker:        _SimWorker | None = None
        self._gerber_folder: Path | None       = None
        self._material_set:  str               = "Set A"
        self._tg_c:          float             = 135.0
        self._board_w_mm:    float             = 0.0
        self._board_h_mm:    float             = 0.0
        self._build()
        _default = Path(r"C:\Users\Asa Guest\Documents\Projects\CopperBalancingApplication\Cu_Balancing_890K_StackUp.json")
        if _default.exists():
            self._load_stackup(_default)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---- Left panel: controls ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 4, 10)
        left_layout.setSpacing(8)

        left_layout.addWidget(self._build_stackup_group())
        left_layout.addWidget(self._build_params_group())

        self._run_btn = QPushButton("Run Simulation")
        self._run_btn.setEnabled(False)
        self._run_btn.setFixedHeight(32)
        self._run_btn.clicked.connect(self._run)
        left_layout.addWidget(self._run_btn)

        left_layout.addWidget(self._build_log_group(), stretch=1)
        left_layout.addWidget(self._build_results_group())

        splitter.addWidget(left)

        # ---- Right panel: heatmap ----
        self._heatmap = HeatmapView()
        splitter.addWidget(self._heatmap)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setHandleWidth(4)

        root.addWidget(splitter)

    def _build_stackup_group(self) -> QGroupBox:
        grp    = QGroupBox("Stackup")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        # File picker
        row = QHBoxLayout()
        self._stackup_path = QLineEdit()
        self._stackup_path.setPlaceholderText("Select a stackup JSON file...")
        self._stackup_path.setReadOnly(True)
        row.addWidget(self._stackup_path)
        browse_btn = QPushButton("Browse...")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_stackup)
        row.addWidget(browse_btn)
        layout.addLayout(row)

        # Board name
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Board name:"))
        self._board_name = QLineEdit()
        self._board_name.setText("Cu_Bal_TV_Q1")
        self._board_name.setFixedWidth(200)
        self._board_name.textChanged.connect(self._update_npz_status)
        self._board_name.textChanged.connect(self._update_run_btn)
        name_row.addWidget(self._board_name)
        name_row.addStretch()
        layout.addLayout(name_row)

        # Info row
        info = QHBoxLayout()
        self._lbl_mat_set = QLabel("Material set: —")
        self._lbl_tg      = QLabel("Tg: —")
        self._lbl_dims    = QLabel("Canvas: —")
        self._lbl_npz     = QLabel("NPZ: —")
        for lbl in (self._lbl_mat_set, self._lbl_tg, self._lbl_dims, self._lbl_npz):
            info.addWidget(lbl)
        info.addStretch()
        layout.addLayout(info)

        return grp

    def _build_params_group(self) -> QGroupBox:
        grp    = QGroupBox("Process Parameters")
        layout = QHBoxLayout(grp)
        layout.setSpacing(20)

        # Temperature ramp inputs
        temp_form = QFormLayout()
        temp_form.setSpacing(6)

        self._spin_T_start = QSpinBox()
        self._spin_T_start.setRange(-50, 500)
        self._spin_T_start.setValue(25)
        self._spin_T_start.setSuffix(" °C")
        temp_form.addRow("Start / ambient:", self._spin_T_start)

        self._spin_T_peak = QSpinBox()
        self._spin_T_peak.setRange(0, 500)
        self._spin_T_peak.setValue(190)
        self._spin_T_peak.setSuffix(" °C")
        temp_form.addRow("Peak (press):", self._spin_T_peak)

        self._spin_rate = QDoubleSpinBox()
        self._spin_rate.setRange(0.1, 20.0)
        self._spin_rate.setValue(3.0)
        self._spin_rate.setSuffix(" °C/s")
        self._spin_rate.setDecimals(1)
        self._spin_rate.setToolTip("Ramp rate is recorded but does not affect the static CLT result.")
        temp_form.addRow("Ramp rate:", self._spin_rate)

        self._spin_hold = QSpinBox()
        self._spin_hold.setRange(0, 300)
        self._spin_hold.setValue(30)
        self._spin_hold.setSuffix(" min")
        self._spin_hold.setToolTip("Hold time is recorded but does not affect the static CLT result.")
        temp_form.addRow("Hold time:", self._spin_hold)

        layout.addLayout(temp_form)

        # DPI
        dpi_form = QFormLayout()
        dpi_form.setSpacing(6)

        self._spin_dpi = QSpinBox()
        self._spin_dpi.setRange(10, 2000)
        self._spin_dpi.setValue(50)
        self._spin_dpi.setSuffix(" DPI")
        self._spin_dpi.setFixedWidth(110)
        self._spin_dpi.valueChanged.connect(self._update_npz_status)
        dpi_form.addRow("Resolution:", self._spin_dpi)

        layout.addLayout(dpi_form)
        layout.addStretch()

        return grp

    def _build_log_group(self) -> QGroupBox:
        grp    = QGroupBox("Log")
        layout = QVBoxLayout(grp)
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(500)
        layout.addWidget(self._log)
        return grp

    def _build_results_group(self) -> QGroupBox:
        grp    = QGroupBox("Results")
        layout = QHBoxLayout(grp)
        self._res_bow   = QLabel("Peak bow: —")
        self._res_ratio = QLabel("Bow/span ratio: —")
        self._res_bwarn = QLabel("B-matrix: —")
        for lbl in (self._res_bow, self._res_ratio, self._res_bwarn):
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl)
        return grp

    # ------------------------------------------------------------------
    # Stackup loading
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

        # Material set and Tg
        self._material_set = data.get("material_set", "Set A")
        self._tg_c         = self._read_tg(self._material_set)
        self._lbl_mat_set.setText(f"Material set: {self._material_set}")
        self._lbl_tg.setText(f"Tg: {self._tg_c:.0f} °C")

        # Gerber folder from first non-null gerber_path in stackup
        self._gerber_folder = None
        for row in data.get("stackup", []):
            gp = row.get("gerber_path")
            if gp:
                self._gerber_folder = Path(gp).parent
                break

        self._update_canvas_dims()
        self._update_npz_status()
        self._update_run_btn()

    def _read_tg(self, set_name: str) -> float:
        try:
            lib = json.loads(_LIBRARY_PATH.read_text(encoding="utf-8"))
            for ms in lib.get("material_sets", []):
                if ms["name"] == set_name:
                    for m in ms.get("materials", []):
                        if m["name"] == "FR4" and m.get("tg_c"):
                            return float(m["tg_c"])
        except Exception:
            pass
        return 135.0

    def _update_canvas_dims(self):
        if not self._gerber_folder:
            return
        try:
            from src.ingestion.gerber_parser import _parse_gerber_extents
            outline = self._gerber_folder / "outline.gbr"
            if outline.exists():
                extents = _parse_gerber_extents(outline)
                if extents:
                    x_min, y_min, x_max, y_max = extents
                    self._board_w_mm = x_max - x_min
                    self._board_h_mm = y_max - y_min
                    self._lbl_dims.setText(
                        f"Canvas: {self._board_w_mm:.0f} x {self._board_h_mm:.0f} mm"
                    )
                    return
        except Exception:
            pass
        self._lbl_dims.setText("Canvas: outline not found")

    def _update_npz_status(self):
        if not self._gerber_folder:
            return
        dpi        = self._spin_dpi.value()
        board_name = self._board_name.text().strip()
        folder_name = f"{board_name}_{dpi}dpi" if board_name else None

        if folder_name and _NPZ_ROOT.exists():
            exact = _NPZ_ROOT / folder_name
            if exact.is_dir() and any(exact.glob("*.npz")):
                count = len(list(exact.glob("*.npz")))
                self._lbl_npz.setText(f"NPZ: {folder_name} ({count} files) — ready")
                return

        label = f"NPZ: {folder_name} — will generate on run" if folder_name else "NPZ: enter board name to resolve"
        self._lbl_npz.setText(label)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def _update_run_btn(self):
        can_run = (
            bool(self._stackup_path.text())
            and self._gerber_folder is not None
            and bool(self._board_name.text().strip())
        )
        self._run_btn.setEnabled(can_run)

    def _run(self):
        if self._thread and self._thread.isRunning():
            return

        stackup_path = self._stackup_path.text()
        board_name   = self._board_name.text().strip()
        if not stackup_path or not self._gerber_folder or not board_name:
            return

        self._log.clear()
        self._res_bow.setText("Peak bow: —")
        self._res_ratio.setText("Bow/span ratio: —")
        self._res_bwarn.setText("B-matrix: —")
        self._run_btn.setEnabled(False)

        params = {
            "stackup_path": stackup_path,
            "gerber_folder": str(self._gerber_folder),
            "board_name":    board_name,
            "dpi":           self._spin_dpi.value(),
            "T_start":       float(self._spin_T_start.value()),
            "T_peak":        float(self._spin_T_peak.value()),
            "T_vit":         self._tg_c,
            "material_set":  self._material_set,
        }

        self._worker = _SimWorker(params)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._log_line)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._update_run_btn)

        self._thread.start()

    def _log_line(self, msg: str):
        self._log.appendPlainText(msg)

    def _on_finished(self, result):
        self._res_bow.setText(f"Peak bow: {result.peak_bow_mm:.4f} mm")
        self._res_ratio.setText(f"Bow/span ratio: {result.bow_span_ratio:.6f}")
        self._res_bwarn.setText(
            "B-matrix: WARNING (lower bound)" if result.b_matrix_warning else "B-matrix: OK"
        )

        ny, nx = result.w_mm.shape
        x_mm = np.linspace(0, self._board_w_mm, nx)
        y_mm = np.linspace(0, self._board_h_mm, ny)
        self._heatmap.show_heatmap(
            z=result.w_mm,
            x=x_mm,
            y=y_mm,
            title="Warpage w (mm)",
        )

        # Emit SimResult so ComparePage can receive it.
        # Pixel-index coords (0..nx-1, 0..ny-1) are used so that after
        # spatial alignment the measurement lands on the same integer grid.
        sim_result = SimResult(
            mode="clt",
            displacement=result.w_mm.copy(),
            x_coords=np.arange(nx, dtype=float),
            y_coords=np.arange(ny, dtype=float),
        )
        self.sim_result_ready.emit(sim_result)

    def _on_error(self, msg: str):
        self._log_line(f"ERROR:\n{msg}")
