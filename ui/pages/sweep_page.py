from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import json

from src.ingestion.stackup import load_stackup
from src.simulation.sweep import count_sweep_combinations, run_sweep

_PREVIEW_COLS = [
    ("Name",        "name",           "s",    "Config entry name (e.g. ACCL-EM890K_Q1). Set from the batch config file."),
    ("DAT File",    "dat_file",       "s",    "Akrometrix .dat filename stem used for this measurement."),
    ("Side",        "side",           "s",    "Board side measured: Top or Bottom, parsed from the .dat filename."),
    ("Material",    "material",       "s",    "Stackup material set name (e.g. EM890K, IT988). Comes from the stackup JSON."),
    ("DPI",         "dpi",            "d",    "Gerber rasterisation resolution. Higher DPI captures finer copper detail but increases memory and solve time."),
    ("ΔT (°C)",     "delta_t_c",      ".1f",  "Temperature delta for CLT solve. 0.0 for imbalance and density tracks (ΔT-independent)."),
    ("Source",      "source",         "s",    "'imbalance' = copper imbalance map (top−bottom weighted by oz and z-position).\n'density'   = total copper density map (unsigned sum of all layers).\n'clt'       = CLT warpage prediction."),
    ("Blur",        "blur_type",      "s",    "'none'     = no blur applied to source map.\n'gaussian' = Gaussian blur applied to source map.\n'box'      = box (uniform) blur applied to source map.\nNot applicable for CLT track (always none)."),
    ("Sigma",       "sigma",          ".1f",  "Gaussian blur standard deviation in pixels. Only relevant when Blur = gaussian."),
    ("Fill",        "fill_missing",   "s",    "Whether biharmonic inpainting was applied to fill gaps in the Akro measurement before comparison."),
    ("Den. σ",      "denoise_sigma",  ".1f",  "Gaussian denoise sigma applied to the Akro measurement before comparison (0 = no denoise)."),
    ("Crop",        "crop_fraction",  ".2f",  "Central fraction of the board used for comparison (1.0 = full board). Cropping excludes edge effects that may not be well-modelled."),
    ("RMSE Z (mm)", "rmse_z",         ".4f",  "Root mean square of the vertical (Z) displacement difference between CLT and Akro. Primary error metric — lower is better."),
    ("MAE Z (mm)",  "mae_z",          ".4f",  "Mean absolute vertical error. Less sensitive to large outliers than RMSE. Lower is better."),
    ("Pearson r",   "pearson_r",      ".4f",  "Pearson correlation coefficient between CLT and Akro displacement fields (−1 to 1). Measures shape agreement independent of scale. Closer to 1 is better."),
    ("R²",          "r2",             ".4f",  "Coefficient of determination — fraction of Akro variance explained by the CLT prediction (0 to 1). Closer to 1 is better."),
    ("Angle mean°", "angle_mean_deg", ".2f",  "Mean angular difference between CLT and Akro gradient vectors (degrees). Measures whether the slope directions agree. Lower is better."),
    ("Mag ratio",   "mag_ratio_mean", ".3f",  "Mean ratio of CLT gradient magnitude to Akro gradient magnitude. 1.0 = perfect amplitude match. <1 = CLT under-predicts local slopes, >1 = over-predicts."),
]


class _SweepWorker(QThread):
    progress = pyqtSignal(int, int)   # completed, total
    finished = pyqtSignal(object)     # pd.DataFrame on success
    error = pyqtSignal(str)

    def __init__(self, configs: list[dict], sweep_params: dict, grand_total: int = 0):
        """
        configs      : list of {"name", "akro_folder", "gerber_folder", "stackup"} dicts.
        sweep_params : shared sweep settings (dpi_values, delta_t_values, etc.).
        grand_total  : pre-computed total combinations across all configs for the progress denominator.
        """
        super().__init__()
        self._configs = configs
        self._p = sweep_params
        self._grand_total = grand_total

    def run(self):
        try:
            all_frames = []
            completed_offset = 0
            for cfg in self._configs:
                last_total = [0]
                offset = completed_offset

                def _cb(done, total, _offset=offset, _lt=last_total):
                    _lt[0] = total
                    denom = self._grand_total if self._grand_total else total
                    self.progress.emit(_offset + done, denom)

                df = run_sweep(
                    akro_folder=cfg["akro_folder"],
                    gerber_folder=cfg["gerber_folder"],
                    stackup=cfg["stackup"],
                    delta_t_values=self._p["delta_t_values"],
                    dpi_values=self._p["dpi_values"],
                    sigma_values=self._p["sigma_values"],
                    box_radius_values=self._p["box_radius_values"],
                    crop_center_fractions=self._p["crop_center_fractions"],
                    fill_missing_values=self._p["fill_missing_values"],
                    denoise_sigma_values=self._p["denoise_sigma_values"],
                    board_name=cfg["name"],
                    cache_name=cfg.get("cache_name", ""),
                    csv_path=cfg.get("csv_path"),
                    progress_cb=_cb,
                )
                all_frames.append(df)
                completed_offset += last_total[0]
            combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
            self.finished.emit(combined)
        except NotImplementedError:
            self.error.emit(
                "CLT solver is not yet implemented.\n"
                "The sweep infrastructure is ready — implement solve_clt() in "
                "src/simulation/clt_solver.py to run this sweep."
            )
        except Exception as exc:
            self.error.emit(str(exc))


class SweepPage(QWidget):
    """DPI × ΔT parameter sweep: CLT simulation vs. Akrometrix folder."""

    _DEFAULT_CONFIG = (
        Path(__file__).parents[2]
        / "assets" / "Cu_Bal_TV_configs.json"
    )

    def __init__(self):
        super().__init__()
        self._results: pd.DataFrame | None = None
        self._worker: _SweepWorker | None = None
        self._dashboard_running: bool = False
        self._build()
        if self._DEFAULT_CONFIG.exists():
            self._config_path.setText(str(self._DEFAULT_CONFIG))
            self._load_configs()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Vertical)

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(6)
        top_layout.addWidget(self._build_paths_group())
        top_layout.addWidget(self._build_params_group())
        top_layout.addWidget(self._build_run_row())
        top.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        splitter.addWidget(top)

        self._table = self._build_table()
        splitter.addWidget(self._table)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    def _build_paths_group(self) -> QGroupBox:
        box = QGroupBox("Inputs")
        form = QFormLayout(box)
        form.setHorizontalSpacing(8)

        # Config file — optional batch import
        cfg_row = QHBoxLayout()
        self._config_path = QLineEdit()
        self._config_path.setReadOnly(True)
        self._config_path.setPlaceholderText("(optional) sweep_configs.json")
        cfg_row.addWidget(self._config_path)
        cfg_browse = QPushButton("Browse")
        cfg_browse.setFixedWidth(70)
        cfg_browse.clicked.connect(self._browse_config)
        cfg_row.addWidget(cfg_browse)
        cfg_load = QPushButton("Load")
        cfg_load.setFixedWidth(55)
        cfg_load.clicked.connect(self._load_configs)
        cfg_row.addWidget(cfg_load)
        form.addRow("Config file:", cfg_row)

        self._config_combo = QComboBox()
        self._config_combo.setPlaceholderText("— select config to fill fields —")
        self._config_combo.currentIndexChanged.connect(self._apply_config)
        form.addRow("Active config:", self._config_combo)

        self._loaded_configs: list[dict] = []

        # Board name — used to key PNG/NPZ cache folders
        self._board_name = QLineEdit()
        self._board_name.setPlaceholderText("e.g. Cu_Bal_TV  (used to name cached render folders)")
        self._board_name.setText("Cu_Bal_TV")
        self._board_name.textChanged.connect(self._update_csv_default)
        form.addRow("Board name:", self._board_name)

        self._csv_path = QLineEdit()
        self._csv_path.setPlaceholderText("(auto-generated from board name)")
        self._update_csv_default()
        form.addRow("Results CSV:", self._csv_path)

        self._akro_path = self._path_row(form, "Akrometrix folder:", folder=True)
        self._akro_path.setText(
            r"C:\Users\Asa Guest\Documents\Projects\CopperBalancingApplication\assets\AkroFiles\TopDatFiles\ACCL-EM890K\Q1"
        )
        self._stackup_path = self._path_row(form, "Stackup JSON:", folder=False,
                                             filter="Stackup Files (*.json);;All Files (*)")
        # Gerber folder label — read-only, auto-resolved from stackup
        self._gerber_label = QLabel("(load a stackup to resolve)")
        self._gerber_label.setStyleSheet("color: #888; font-style: italic;")
        form.addRow("Gerber folder:", self._gerber_label)
        self._stackup_path.textChanged.connect(self._resolve_gerber_from_stackup)
        self._stackup_path.setText(
            r"C:\Users\Asa Guest\Documents\Projects\CopperBalancingApplication\Cu_Balancing_890K_StackUp.json"
        )
        return box

    def _path_row(self, form: QFormLayout, label: str,
                  folder: bool, filter: str = "") -> QLineEdit:
        row = QHBoxLayout()
        edit = QLineEdit()
        edit.setReadOnly(True)
        edit.setPlaceholderText("(not set)")
        row.addWidget(edit)
        btn = QPushButton("Browse")
        btn.setFixedWidth(70)
        btn.clicked.connect(lambda _, e=edit, f=folder, ft=filter: self._browse(e, f, ft))
        row.addWidget(btn)
        form.addRow(label, row)
        return edit

    def _resolve_gerber_from_stackup(self) -> None:
        path_str = self._stackup_path.text().strip()
        if not path_str:
            self._gerber_label.setText("(load a stackup to resolve)")
            self._gerber_label.setStyleSheet("color: #888; font-style: italic;")
            return
        # Auto-fill board name from the stackup filename stem if the field is empty
        if not self._board_name.text().strip():
            self._board_name.setText(Path(path_str).stem)
        try:
            data = json.loads(Path(path_str).read_text(encoding="utf-8"))
            for row in data.get("stackup", []):
                gp = row.get("gerber_path")
                if gp:
                    folder = Path(gp).parent
                    self._gerber_label.setText(str(folder))
                    self._gerber_label.setStyleSheet("color: #d4d4d4;")
                    return
            self._gerber_label.setText("No Gerber paths found in stackup")
            self._gerber_label.setStyleSheet("color: #cc6666;")
        except Exception:
            self._gerber_label.setText("Could not read stackup")
            self._gerber_label.setStyleSheet("color: #cc6666;")

    def _update_csv_default(self) -> None:
        name = self._board_name.text().strip() or "sweep"
        from pathlib import Path as _P
        default = str(
            _P(__file__).parents[2] / "assets" / "sweep_results" / f"{name}_sweep.csv"
        )
        self._csv_path.setPlaceholderText(default)
        # Only overwrite if it still matches the previous auto-generated value
        cur = self._csv_path.text().strip()
        if not cur or cur.endswith("_sweep.csv"):
            self._csv_path.setText(default)

    def _browse(self, edit: QLineEdit, folder: bool, file_filter: str) -> None:
        if folder:
            path = QFileDialog.getExistingDirectory(self, "Select Folder")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Open File", "", file_filter)
        if path:
            edit.setText(path)

    def _build_params_group(self) -> QGroupBox:
        box = QGroupBox("Sweep Parameters")
        outer = QVBoxLayout(box)
        outer.setSpacing(4)

        def _preview_label() -> QLabel:
            lbl = QLabel()
            lbl.setStyleSheet("color: #888; font-family: monospace; font-size: 10px;")
            lbl.setContentsMargins(4, 0, 0, 4)
            return lbl

        # ΔT row
        dt_row = QHBoxLayout()
        dt_row.setSpacing(12)
        dt_row.addWidget(QLabel("ΔT  Min:"))
        self._dt_min = QDoubleSpinBox()
        self._dt_min.setRange(0, 400)
        self._dt_min.setValue(100)
        self._dt_min.setSuffix(" °C")
        self._dt_min.setDecimals(1)
        dt_row.addWidget(self._dt_min)
        dt_row.addWidget(QLabel("Max:"))
        self._dt_max = QDoubleSpinBox()
        self._dt_max.setRange(0, 400)
        self._dt_max.setValue(200)
        self._dt_max.setSuffix(" °C")
        self._dt_max.setDecimals(1)
        dt_row.addWidget(self._dt_max)
        dt_row.addWidget(QLabel("Steps:"))
        self._dt_steps = QSpinBox()
        self._dt_steps.setRange(1, 200)
        self._dt_steps.setValue(2)
        dt_row.addWidget(self._dt_steps)
        dt_row.addStretch()
        outer.addLayout(dt_row)
        self._dt_preview = _preview_label()
        outer.addWidget(self._dt_preview)

        # DPI row
        dpi_row = QHBoxLayout()
        dpi_row.setSpacing(12)
        dpi_row.addWidget(QLabel("DPI  Min:"))
        self._dpi_min = QSpinBox()
        self._dpi_min.setRange(50, 5000)
        self._dpi_min.setValue(50)
        self._dpi_min.setSuffix(" dpi")
        self._dpi_min.setSingleStep(50)
        dpi_row.addWidget(self._dpi_min)
        dpi_row.addWidget(QLabel("Max:"))
        self._dpi_max = QSpinBox()
        self._dpi_max.setRange(50, 5000)
        self._dpi_max.setValue(300)
        self._dpi_max.setSuffix(" dpi")
        self._dpi_max.setSingleStep(50)
        dpi_row.addWidget(self._dpi_max)
        dpi_row.addWidget(QLabel("Steps:"))
        self._dpi_steps = QSpinBox()
        self._dpi_steps.setRange(1, 50)
        self._dpi_steps.setValue(6)
        dpi_row.addWidget(self._dpi_steps)
        dpi_row.addStretch()
        outer.addLayout(dpi_row)
        self._dpi_preview = _preview_label()
        outer.addWidget(self._dpi_preview)

        # Gaussian sigma row
        sigma_row = QHBoxLayout()
        sigma_row.setSpacing(12)
        self._gauss_cb = QCheckBox("Gaussian σ")
        self._gauss_cb.setChecked(True)
        self._gauss_cb.setToolTip("Sweep Gaussian blur applied to the copper imbalance map. σ = 0 adds a raw density baseline row.")
        self._gauss_cb.toggled.connect(self._on_gauss_toggled)
        sigma_row.addWidget(self._gauss_cb)
        sigma_row.addWidget(QLabel("Min:"))
        self._sigma_min = QDoubleSpinBox()
        self._sigma_min.setRange(0, 1000)
        self._sigma_min.setValue(0)
        self._sigma_min.setSuffix(" px")
        self._sigma_min.setDecimals(1)
        sigma_row.addWidget(self._sigma_min)
        sigma_row.addWidget(QLabel("Max:"))
        self._sigma_max = QDoubleSpinBox()
        self._sigma_max.setRange(0, 1000)
        self._sigma_max.setValue(500)
        self._sigma_max.setSuffix(" px")
        self._sigma_max.setDecimals(1)
        sigma_row.addWidget(self._sigma_max)
        sigma_row.addWidget(QLabel("Steps:"))
        self._sigma_steps = QSpinBox()
        self._sigma_steps.setRange(1, 50)
        self._sigma_steps.setValue(6)
        sigma_row.addWidget(self._sigma_steps)
        sigma_row.addStretch()
        outer.addLayout(sigma_row)
        self._sigma_preview = _preview_label()
        outer.addWidget(self._sigma_preview)

        # Box blur radius row
        box_row = QHBoxLayout()
        box_row.setSpacing(12)
        self._box_cb = QCheckBox("Box radius")
        self._box_cb.setChecked(True)
        self._box_cb.setToolTip("Include box (uniform) blur rows in the sweep output.")
        self._box_cb.toggled.connect(self._on_box_toggled)
        box_row.addWidget(self._box_cb)
        box_row.addWidget(QLabel("Min:"))
        self._box_min = QDoubleSpinBox()
        self._box_min.setRange(0, 1000)
        self._box_min.setValue(10)
        self._box_min.setSuffix(" px")
        self._box_min.setDecimals(1)
        box_row.addWidget(self._box_min)
        box_row.addWidget(QLabel("Max:"))
        self._box_max = QDoubleSpinBox()
        self._box_max.setRange(0, 1000)
        self._box_max.setValue(500)
        self._box_max.setSuffix(" px")
        self._box_max.setDecimals(1)
        box_row.addWidget(self._box_max)
        box_row.addWidget(QLabel("Steps:"))
        self._box_steps = QSpinBox()
        self._box_steps.setRange(1, 50)
        self._box_steps.setValue(6)
        box_row.addWidget(self._box_steps)
        box_row.addStretch()
        outer.addLayout(box_row)
        self._box_preview = _preview_label()
        outer.addWidget(self._box_preview)

        # Center crop row
        crop_row = QHBoxLayout()
        crop_row.setSpacing(12)
        crop_row.addWidget(QLabel("Center crop  Min:"))
        self._crop_min = QDoubleSpinBox()
        self._crop_min.setRange(0.1, 1.0)
        self._crop_min.setValue(0.2)
        self._crop_min.setSingleStep(0.1)
        self._crop_min.setDecimals(2)
        self._crop_min.setToolTip("Fraction of each board dimension kept (1.0 = no crop).")
        crop_row.addWidget(self._crop_min)
        crop_row.addWidget(QLabel("Max:"))
        self._crop_max = QDoubleSpinBox()
        self._crop_max.setRange(0.1, 1.0)
        self._crop_max.setValue(0.7)
        self._crop_max.setSingleStep(0.1)
        self._crop_max.setDecimals(2)
        crop_row.addWidget(self._crop_max)
        crop_row.addWidget(QLabel("Steps:"))
        self._crop_steps = QSpinBox()
        self._crop_steps.setRange(1, 10)
        self._crop_steps.setValue(6)
        crop_row.addWidget(self._crop_steps)
        crop_row.addStretch()
        outer.addLayout(crop_row)
        self._crop_preview = _preview_label()
        outer.addWidget(self._crop_preview)

        # Preprocess row
        pre_row = QHBoxLayout()
        pre_row.setSpacing(12)
        self._fill_sweep_cb = QCheckBox("Sweep fill missing (both on/off)")
        self._fill_sweep_cb.setChecked(True)
        self._fill_sweep_cb.setToolTip(
            "When checked, the sweep runs each combination both with and without\n"
            "biharmonic inpainting applied to the Akro measurement."
        )
        pre_row.addWidget(self._fill_sweep_cb)
        self._denoise_sweep_cb = QCheckBox("Denoise σ")
        self._denoise_sweep_cb.setChecked(True)
        self._denoise_sweep_cb.setToolTip("Sweep Gaussian denoise sigma. σ = 0 includes a no-denoise baseline.")
        self._denoise_sweep_cb.toggled.connect(self._on_denoise_toggled)
        pre_row.addWidget(self._denoise_sweep_cb)
        pre_row.addWidget(QLabel("Min:"))
        self._denoise_min = QDoubleSpinBox()
        self._denoise_min.setRange(0, 100)
        self._denoise_min.setValue(0)
        self._denoise_min.setSuffix(" px")
        self._denoise_min.setDecimals(1)
        pre_row.addWidget(self._denoise_min)
        pre_row.addWidget(QLabel("Max:"))
        self._denoise_max = QDoubleSpinBox()
        self._denoise_max.setRange(0, 100)
        self._denoise_max.setValue(100)
        self._denoise_max.setSuffix(" px")
        self._denoise_max.setDecimals(1)
        pre_row.addWidget(self._denoise_max)
        pre_row.addWidget(QLabel("Steps:"))
        self._denoise_steps = QSpinBox()
        self._denoise_steps.setRange(1, 20)
        self._denoise_steps.setValue(5)
        pre_row.addWidget(self._denoise_steps)
        pre_row.addStretch()
        outer.addLayout(pre_row)
        self._denoise_preview = _preview_label()
        outer.addWidget(self._denoise_preview)

        # Wire all spinboxes to the preview updater
        for w in (self._dt_min, self._dt_max, self._dt_steps,
                  self._dpi_min, self._dpi_max, self._dpi_steps,
                  self._sigma_min, self._sigma_max, self._sigma_steps,
                  self._box_min, self._box_max, self._box_steps,
                  self._crop_min, self._crop_max, self._crop_steps,
                  self._denoise_min, self._denoise_max, self._denoise_steps):
            w.valueChanged.connect(self._update_param_previews)
        self._update_param_previews()

        return box

    @staticmethod
    def _fmt_values(values: list, fmt: str = ".1f", n: int = 5) -> str:
        if not values:
            return "—"
        strs = [format(v, fmt) for v in values]
        if len(strs) <= 2 * n:
            return "  ".join(strs)
        return "  ".join(strs[:n]) + "  …  " + "  ".join(strs[-n:])

    def _update_param_previews(self) -> None:
        def _linspace(lo, hi, steps):
            if steps <= 1:
                return [lo]
            return [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]

        dt  = _linspace(self._dt_min.value(), self._dt_max.value(), self._dt_steps.value())
        dpi = [round(v) for v in _linspace(self._dpi_min.value(), self._dpi_max.value(), self._dpi_steps.value())]
        sig = _linspace(self._sigma_min.value(), self._sigma_max.value(), self._sigma_steps.value())
        box = _linspace(self._box_min.value(), self._box_max.value(), self._box_steps.value())
        crp = _linspace(self._crop_min.value(), self._crop_max.value(), self._crop_steps.value())
        den = _linspace(self._denoise_min.value(), self._denoise_max.value(), self._denoise_steps.value())

        self._dt_preview.setText("  →  " + self._fmt_values(dt,  ".1f") + " °C")
        self._dpi_preview.setText("  →  " + self._fmt_values(dpi, ".0f") + " dpi")
        self._sigma_preview.setText("  →  " + self._fmt_values(sig, ".1f") + " px")
        self._box_preview.setText("  →  " + self._fmt_values(box, ".1f") + " px")
        self._crop_preview.setText("  →  " + self._fmt_values(crp, ".2f"))
        self._denoise_preview.setText("  →  " + self._fmt_values(den, ".1f") + " px")

    def _build_run_row(self) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self._run_btn = QPushButton("Run Sweep")
        self._run_btn.setFixedWidth(110)
        self._run_btn.clicked.connect(self._run)
        row.addWidget(self._run_btn)

        self._batch_btn = QPushButton("Run Batch")
        self._batch_btn.setFixedWidth(100)
        self._batch_btn.setToolTip("Run the sweep for every config in the loaded config file.")
        self._batch_btn.clicked.connect(self._run_batch)
        row.addWidget(self._batch_btn)

        self._dashboard_btn = QPushButton("Launch Dashboard")
        self._dashboard_btn.setFixedWidth(140)
        self._dashboard_btn.setToolTip("Start the Dash analysis dashboard and open it in your browser.")
        self._dashboard_btn.clicked.connect(self._launch_dashboard)
        row.addWidget(self._dashboard_btn)

        self._load_csv_btn = QPushButton("Load CSV…")
        self._load_csv_btn.setFixedWidth(90)
        self._load_csv_btn.clicked.connect(self._load_csv)
        row.addWidget(self._load_csv_btn)

        self._save_btn = QPushButton("Save CSV…")
        self._save_btn.setFixedWidth(90)
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_csv)
        row.addWidget(self._save_btn)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        self._progress.setTextVisible(True)
        row.addWidget(self._progress, stretch=1)

        self._status = QLabel("")
        row.addWidget(self._status)

        return w

    def _build_table(self) -> QTableWidget:
        t = QTableWidget(0, len(_PREVIEW_COLS) + 1)   # +1 for View Fit button
        t.setHorizontalHeaderLabels([c[0] for c in _PREVIEW_COLS] + [""])
        for col, (_, _, _, tip) in enumerate(_PREVIEW_COLS):
            t.horizontalHeaderItem(col).setToolTip(tip)
        hdr = t.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(len(_PREVIEW_COLS), QHeaderView.ResizeMode.Fixed)
        t.setColumnWidth(len(_PREVIEW_COLS), 80)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        return t

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _browse_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Config File", "", "JSON Files (*.json);;All Files (*)"
        )
        if path:
            self._config_path.setText(path)
            self._load_configs()

    def _load_configs(self) -> None:
        path_str = self._config_path.text().strip()
        if not path_str:
            return
        try:
            data = json.loads(Path(path_str).read_text(encoding="utf-8"))
            self._loaded_configs = data.get("configs", [])
            self._config_combo.clear()
            for cfg in self._loaded_configs:
                self._config_combo.addItem(cfg.get("name", "(unnamed)"))
            if self._loaded_configs:
                self._config_combo.setCurrentIndex(0)

            # Derive the board name and CSV path from the config file stem
            _results = Path(__file__).parents[2] / "assets" / "sweep_results"
            stem = Path(path_str).stem
            for _sfx in ("_configs", "_config", "_sweep"):
                if stem.endswith(_sfx):
                    stem = stem[: -len(_sfx)]
                    break
            # Board name = design name shared by all entries in this config file
            self._board_name.setText(stem)
            default_csv = str(_results / f"{stem}.csv")
            if not self._csv_path.text().strip():
                self._csv_path.setText(default_csv)
        except Exception as exc:
            QMessageBox.critical(self, "Config load failed", str(exc))

    def _apply_config(self, index: int) -> None:
        if index < 0 or index >= len(self._loaded_configs):
            return
        cfg = self._loaded_configs[index]
        # Board name stays as the config FILE stem — don't overwrite it per entry
        if akro := cfg.get("akro_folder"):
            self._akro_path.setText(akro)
        if stackup := cfg.get("stackup"):
            self._stackup_path.setText(stackup)

    def _on_gauss_toggled(self, checked: bool) -> None:
        for w in (self._sigma_min, self._sigma_max, self._sigma_steps):
            w.setEnabled(checked)

    def _on_box_toggled(self, checked: bool) -> None:
        for w in (self._box_min, self._box_max, self._box_steps):
            w.setEnabled(checked)

    def _on_denoise_toggled(self, checked: bool) -> None:
        for w in (self._denoise_min, self._denoise_max, self._denoise_steps):
            w.setEnabled(checked)

    # ------------------------------------------------------------------
    # Sweep param helpers
    # ------------------------------------------------------------------

    def _build_sweep_params(self) -> dict | None:
        """Validate and return the shared sweep parameter dict, or None on error."""
        dt_min, dt_max = self._dt_min.value(), self._dt_max.value()
        if dt_max < dt_min:
            QMessageBox.warning(self, "Invalid range", "ΔT Max must be ≥ Min.")
            return None
        delta_t_values = list(np.linspace(dt_min, dt_max, self._dt_steps.value()))

        dpi_min, dpi_max = self._dpi_min.value(), self._dpi_max.value()
        if dpi_max < dpi_min:
            QMessageBox.warning(self, "Invalid range", "DPI Max must be ≥ Min.")
            return None
        seen_dpi: set[int] = set()
        dpi_values = [d for d in (int(round(v)) for v in
                      np.linspace(dpi_min, dpi_max, self._dpi_steps.value()))
                      if not (d in seen_dpi or seen_dpi.add(d))]  # type: ignore

        sigma_values: list[float] = []
        if self._gauss_cb.isChecked():
            if self._sigma_max.value() < self._sigma_min.value():
                QMessageBox.warning(self, "Invalid range", "Gaussian σ Max must be ≥ Min.")
                return None
            seen_s: set[float] = set()
            for v in np.linspace(self._sigma_min.value(), self._sigma_max.value(),
                                 self._sigma_steps.value()):
                rv = round(float(v), 2)
                if rv not in seen_s:
                    seen_s.add(rv)
                    sigma_values.append(rv)

        box_radius_values: list[float] = []
        if self._box_cb.isChecked():
            if self._box_max.value() < self._box_min.value():
                QMessageBox.warning(self, "Invalid range", "Box radius Max must be ≥ Min.")
                return None
            seen_b: set[float] = set()
            for v in np.linspace(self._box_min.value(), self._box_max.value(),
                                 self._box_steps.value()):
                rv = round(float(v), 2)
                if rv not in seen_b:
                    seen_b.add(rv)
                    box_radius_values.append(rv)

        if not sigma_values and not box_radius_values:
            QMessageBox.warning(self, "No blur ops", "Enable at least one blur type (Gaussian or Box).")
            return None

        if self._crop_max.value() < self._crop_min.value():
            QMessageBox.warning(self, "Invalid range", "Crop Max must be ≥ Min.")
            return None
        seen_c: set[float] = set()
        crop_fractions = [c for c in (round(v, 2) for v in
                          np.linspace(self._crop_min.value(), self._crop_max.value(),
                                      self._crop_steps.value()))
                          if not (c in seen_c or seen_c.add(c))]  # type: ignore

        fill_missing_values = [False, True] if self._fill_sweep_cb.isChecked() else [False]

        denoise_sigma_values: list[float] = [0.0]
        if self._denoise_sweep_cb.isChecked():
            if self._denoise_max.value() < self._denoise_min.value():
                QMessageBox.warning(self, "Invalid range", "Denoise σ Max must be ≥ Min.")
                return None
            seen_d: set[float] = set()
            for v in np.linspace(self._denoise_min.value(), self._denoise_max.value(),
                                 self._denoise_steps.value()):
                rv = round(float(v), 2)
                if rv not in seen_d:
                    seen_d.add(rv)
                    denoise_sigma_values.append(rv)
            seen_f: set[float] = set()
            denoise_sigma_values = [x for x in denoise_sigma_values
                                    if not (x in seen_f or seen_f.add(x))]  # type: ignore

        return {
            "delta_t_values":      delta_t_values,
            "dpi_values":          dpi_values,
            "sigma_values":        sigma_values,
            "box_radius_values":   box_radius_values,
            "crop_center_fractions": crop_fractions,
            "fill_missing_values": fill_missing_values,
            "denoise_sigma_values": denoise_sigma_values,
        }

    def _resolve_config(self, akro_str: str, stackup_str: str,
                        board_name: str) -> dict | None:
        """Load stackup + resolve gerber folder for one config. Returns config dict or None."""
        try:
            stackup = load_stackup(Path(stackup_str))
        except Exception as exc:
            QMessageBox.critical(self, "Stackup load failed", str(exc))
            return None

        gerber_folder: Path | None = None
        try:
            data = json.loads(Path(stackup_str).read_text(encoding="utf-8"))
            for row in data.get("stackup", []):
                gp = row.get("gerber_path")
                if gp:
                    gerber_folder = Path(gp).parent
                    break
        except Exception:
            pass
        if gerber_folder is None:
            QMessageBox.critical(self, "Missing Gerbers",
                                 f"No gerber_path entries found in {Path(stackup_str).name}.")
            return None

        csv_str = self._csv_path.text().strip()
        return {
            "name":         board_name or gerber_folder.name,
            "akro_folder":  Path(akro_str),
            "gerber_folder": gerber_folder,
            "stackup":      stackup,
            "csv_path":     Path(csv_str) if csv_str else None,
        }

    def _start_worker(self, configs: list[dict], sweep_params: dict,
                      total_combinations: int = 0) -> None:
        self._run_btn.setEnabled(False)
        self._batch_btn.setEnabled(False)
        self._save_btn.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        if total_combinations:
            self._status.setText(
                f"Running {len(configs)} config(s) — {total_combinations:,} total combinations…"
            )
        else:
            self._status.setText(f"Running {len(configs)} config(s)…")

        self._worker = _SweepWorker(configs, sweep_params, grand_total=total_combinations)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    # ------------------------------------------------------------------
    # Run actions
    # ------------------------------------------------------------------

    def _run(self) -> None:
        akro_str    = self._akro_path.text().strip()
        stackup_str = self._stackup_path.text().strip()
        if not akro_str:
            QMessageBox.warning(self, "Missing input", "Select an Akrometrix folder.")
            return
        if not stackup_str:
            QMessageBox.warning(self, "Missing input", "Select a Stackup JSON file.")
            return

        sweep_params = self._build_sweep_params()
        if sweep_params is None:
            return

        cfg = self._resolve_config(akro_str, stackup_str,
                                   self._board_name.text().strip())
        if cfg is None:
            return

        self._start_worker([cfg], sweep_params)

    def _run_batch(self) -> None:
        if not self._loaded_configs:
            QMessageBox.warning(self, "No configs", "Load a config file first.")
            return

        sweep_params = self._build_sweep_params()
        if sweep_params is None:
            return

        # All batch configs share one CSV — named after the config JSON file
        csv_str = self._csv_path.text().strip()
        if not csv_str:
            config_str = self._config_path.text().strip()
            _results = Path(__file__).parents[2] / "assets" / "sweep_results"
            stem = Path(config_str).stem if config_str else "batch"
            for _sfx in ("_configs", "_config", "_sweep"):
                if stem.endswith(_sfx):
                    stem = stem[: -len(_sfx)]
                    break
            csv_str = str(_results / f"{stem}.csv")
            self._csv_path.setText(csv_str)
        shared_csv = Path(csv_str)

        # Top-level board name used for the raster cache folder naming
        cache_name = self._board_name.text().strip()

        configs = []
        total_combinations = 0
        for raw in self._loaded_configs:
            cfg = self._resolve_config(
                raw.get("akro_folder", ""),
                raw.get("stackup", ""),
                raw.get("name", ""),
            )
            if cfg is None:
                return  # error already shown
            cfg["csv_path"]   = shared_csv   # all configs → same CSV
            cfg["cache_name"] = cache_name   # shared raster cache key
            configs.append(cfg)

            n_dat = len(list(cfg["akro_folder"].glob("*.dat")))
            total_combinations += count_sweep_combinations(
                n_dat_files=n_dat,
                dpi_values=sweep_params["dpi_values"],
                sigma_values=sweep_params["sigma_values"],
                box_radius_values=sweep_params["box_radius_values"],
                delta_t_values=sweep_params["delta_t_values"],
                fill_missing_values=sweep_params["fill_missing_values"],
                denoise_sigma_values=sweep_params["denoise_sigma_values"],
                crop_center_fractions=sweep_params["crop_center_fractions"],
            )

        self._start_worker(configs, sweep_params, total_combinations)

    def _launch_dashboard(self) -> None:
        import threading, webbrowser
        from pathlib import Path as _P

        _RESULTS = _P(__file__).parents[2] / "assets" / "sweep_results"
        _URL = "http://127.0.0.1:8050"

        # Already running — just open the browser
        if self._dashboard_running:
            webbrowser.open(_URL)
            return

        try:
            from dashboard import run_dashboard as _run_dash
        except ImportError:
            QMessageBox.critical(self, "Missing dependency",
                                 "Could not import dashboard.\n"
                                 "Install: pip install dash dash-bootstrap-components")
            return

        # Let the user pick which CSV to load
        default_dir = str(_RESULTS) if _RESULTS.exists() else ""
        csv_str, _ = QFileDialog.getOpenFileName(
            self, "Select sweep results CSV", default_dir,
            "CSV Files (*.csv);;All Files (*)"
        )
        if not csv_str:
            return  # user cancelled
        csv = _P(csv_str)

        _err: list[str] = []

        def _serve():
            try:
                _run_dash(data_path=str(csv), debug=False)
            except Exception as exc:
                _err.append(str(exc))

        threading.Thread(target=_serve, daemon=True).start()
        self._dashboard_running = True
        self._dashboard_btn.setText("Open Dashboard")
        self._dashboard_btn.setToolTip(f"Open dashboard in browser ({csv.name})")

        def _check_and_open():
            if _err:
                self._dashboard_running = False
                QMessageBox.critical(self, "Dashboard failed", _err[0])
                return
            webbrowser.open(_URL)

        QTimer.singleShot(1500, _check_and_open)
        self._status.setText(f"Dashboard starting — {csv.name}")

    def _on_progress(self, done: int, total: int) -> None:
        if self._progress.maximum() != total:
            self._progress.setRange(0, total)
        self._progress.setValue(done)
        self._status.setText(f"{done}/{total}")

    def _on_finished(self, df: pd.DataFrame) -> None:
        self._results = df
        self._progress.setVisible(False)
        self._status.setText(f"Done — {len(df)} rows.")
        self._run_btn.setEnabled(True)
        self._batch_btn.setEnabled(True)
        self._save_btn.setEnabled(True)
        self._populate_table(df)

    def _on_error(self, msg: str) -> None:
        self._progress.setVisible(False)
        self._status.setText("Error.")
        self._run_btn.setEnabled(True)
        self._batch_btn.setEnabled(True)
        QMessageBox.critical(self, "Sweep failed", msg)

    def _load_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load sweep results", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        try:
            df = pd.read_csv(path, on_bad_lines="skip", engine="python")
            self._results = df
            self._save_btn.setEnabled(True)
            self._status.setText(f"Loaded {len(df):,} rows from {Path(path).name}")
            self._populate_table(df)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))

    def _save_csv(self) -> None:
        if self._results is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save sweep results", "", "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        if not path.endswith(".csv"):
            path += ".csv"
        try:
            self._results.to_csv(path, index=False)
            self._status.setText(f"Saved → {Path(path).name}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    # ------------------------------------------------------------------
    # Table population
    # ------------------------------------------------------------------

    def _populate_table(self, df: pd.DataFrame) -> None:
        self._table.setRowCount(len(df))
        for row_idx, record in df.iterrows():
            for col_idx, (_, key, fmt, *_) in enumerate(_PREVIEW_COLS):
                val = record[key]
                if fmt == "s":
                    text = str(val)
                elif fmt == "d":
                    text = format(int(val), "d")
                else:
                    try:
                        text = format(float(val), fmt)
                    except (ValueError, TypeError):
                        text = str(val)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(row_idx, col_idx, item)

            btn = QPushButton("View Fit")
            row_data = record.to_dict()
            btn.clicked.connect(lambda _, r=row_data: self._show_fit_viewer(r))
            self._table.setCellWidget(row_idx, len(_PREVIEW_COLS), btn)

    def _show_fit_viewer(self, row: dict) -> None:
        from ui.components.fit_viewer import FitViewerDialog
        stackup_str = self._stackup_path.text().strip()
        hint = Path(stackup_str) if stackup_str else None
        dlg = FitViewerDialog(row, stackup_hint=hint, parent=self)
        dlg.show()
