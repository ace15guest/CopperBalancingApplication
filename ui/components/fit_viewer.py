"""3-D surface fit viewer — overlays simulation and Akrometrix surfaces in Plotly."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import os
import tempfile

from PyQt6.QtCore import QThread, QUrl, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout,
)

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    _HAS_WEBENGINE = True
except ImportError:
    _HAS_WEBENGINE = False

_PROJECT_ROOT = Path(__file__).parents[2]
_NPZ_ROOT     = _PROJECT_ROOT / "assets" / "processed_npz"
_AKRO_ROOT    = _PROJECT_ROOT / "assets" / "AkroFiles"
_STACKUP_ROOT = _PROJECT_ROOT / "assets" / "StackUps"


# ---------------------------------------------------------------------------
# File-resolution helpers
# ---------------------------------------------------------------------------

def _find_dat(name: str) -> Path:
    candidates = list(_AKRO_ROOT.rglob(f"{name}.dat"))
    if not candidates:
        raise FileNotFoundError(
            f"No .dat file found for '{name}' under {_AKRO_ROOT}.\n"
            "Ensure AkroFiles are present locally."
        )
    return candidates[0]


def _find_npz_folder(dpi: int, material: str) -> Path:
    if not _NPZ_ROOT.exists():
        raise FileNotFoundError(f"NPZ cache folder not found: {_NPZ_ROOT}")
    candidates = [
        d for d in _NPZ_ROOT.iterdir()
        if d.is_dir() and d.name.endswith(f"_{dpi}dpi")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No cached NPZ folder found for DPI={dpi}.\n"
            "Run the sweep at this DPI first to generate the raster cache."
        )
    # Prefer a folder whose name contains the material set name
    for c in candidates:
        if material.lower() in c.name.lower():
            return c
    return candidates[0]


def _find_stackup(material: str, hint: Path | None) -> Any:
    from src.ingestion.stackup import load_stackup

    # 1. Explicit hint that matches
    if hint and hint.exists():
        s = load_stackup(hint)
        if s.material_set_name == material:
            return s

    # 2. Search StackUps folder by "material_set" key in JSON
    for p in _STACKUP_ROOT.rglob("*.json"):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if raw.get("material_set", "") == material:
                return load_stackup(p)
        except Exception:
            continue

    # 3. Single-stackup fallback
    all_jsons = list(_STACKUP_ROOT.rglob("*.json"))
    if len(all_jsons) == 1:
        return load_stackup(all_jsons[0])

    raise FileNotFoundError(
        f"No stackup JSON found for material '{material}'.\n"
        "Load the matching stackup in the Sweep page first."
    )


def _load_density_maps(npz_folder: Path, stackup: Any) -> list:
    from src.models import CopperDensityMap

    npz_by_stem = {p.stem: p for p in sorted(npz_folder.glob("*.npz"))}
    density_maps: list[CopperDensityMap] = []
    for row in stackup.rows:
        if row.row_type != "copper" or row.gerber_path is None:
            continue
        stem = Path(row.gerber_path).stem
        npz_path = npz_by_stem.get(stem)
        if npz_path is None:
            continue
        data = np.load(npz_path)
        density_maps.append(CopperDensityMap(
            layer_name=f"L{row.layer_number}",
            density=data["density"].astype(np.float32),
            x_coords=np.arange(data["density"].shape[1], dtype=np.float32),
            y_coords=np.arange(data["density"].shape[0], dtype=np.float32),
        ))
    return density_maps


# ---------------------------------------------------------------------------
# Sim-array reconstruction
# ---------------------------------------------------------------------------

def _build_sim_array(row: dict, density_maps: list, stackup: Any,
                     pixel_size_m: float) -> np.ndarray:
    from src.simulation.sweep import (
        _apply_blur, _compute_density_map, _compute_imbalance_map,
    )

    source    = str(row["source"])
    blur_type = str(row["blur_type"])
    sigma     = float(row["sigma"])
    radius    = float(row["radius"])
    delta_t   = float(row["delta_t_c"])

    if source == "imbalance":
        arr = _compute_imbalance_map(density_maps, stackup)
    elif source == "density":
        arr = _compute_density_map(density_maps, stackup)
    elif source == "clt":
        from src.simulation.clt_solver import solve_clt
        result = solve_clt(stackup, density_maps,
                           delta_temp_c=delta_t, pixel_size_m=pixel_size_m)
        arr = result.displacement
    else:
        raise ValueError(f"Unknown source: {source!r}")

    return _apply_blur(arr, blur_type, radius, sigma)


def _center_crop(arr: np.ndarray, fraction: float) -> np.ndarray:
    if fraction >= 1.0:
        return arr
    ny, nx = arr.shape
    dy = int(ny * (1.0 - fraction) / 2)
    dx = int(nx * (1.0 - fraction) / 2)
    return arr[dy: ny - dy, dx: nx - dx]


def _fill_nan_for_display(arr: np.ndarray) -> np.ndarray:
    """Replace NaN with nearest valid value — display only, does not affect metrics."""
    from scipy.ndimage import distance_transform_edt
    nan_mask = ~np.isfinite(arr)
    if not nan_mask.any():
        return arr
    _, ind = distance_transform_edt(nan_mask, return_indices=True)
    out = arr.copy()
    out[nan_mask] = arr[ind[0][nan_mask], ind[1][nan_mask]]
    return out


# ---------------------------------------------------------------------------
# HTML builder (runs in worker thread)
# ---------------------------------------------------------------------------

def _parse_bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def _build_html(row: dict, stackup_hint: Path | None) -> str:
    import plotly.graph_objects as go

    from src.analysis.alignment import align
    from src.ingestion.akrometrix import load_akrometrix_dat
    from src.processing.denoise import denoise as _denoise
    from src.processing.inpainting import fill_missing as _fill_missing

    dat_file_stem = str(row.get("dat_file") or row["name"])
    dpi           = int(row["dpi"])
    material      = str(row["material"])
    fill_missing  = _parse_bool(row["fill_missing"])
    denoise_sigma = float(row["denoise_sigma"])
    crop_fraction = float(row["crop_fraction"])

    # --- Measurement: prefer stored akro_folder path, fall back to rglob ---
    stored_akro = row.get("akro_folder", "")
    if stored_akro and Path(stored_akro).is_dir():
        candidates = list(Path(stored_akro).glob(f"{dat_file_stem}.dat"))
        if not candidates:
            candidates = list(_find_dat(dat_file_stem).parent.glob(f"{dat_file_stem}.dat"))
        dat_path = candidates[0] if candidates else _find_dat(dat_file_stem)
    else:
        dat_path = _find_dat(dat_file_stem)
    meas = load_akrometrix_dat(dat_path)
    if fill_missing:
        meas = _fill_missing(meas)
    if denoise_sigma > 0:
        meas = _denoise(meas, sigma=denoise_sigma)

    # --- Simulation ---
    stackup      = _find_stackup(material, stackup_hint)
    npz_folder   = _find_npz_folder(dpi, material)
    density_maps = _load_density_maps(npz_folder, stackup)
    pixel_size_m = 25.4e-3 / dpi
    sim_arr      = _build_sim_array(row, density_maps, stackup, pixel_size_m)

    # --- Align measurement onto sim grid ---
    aligned_meas, params = align(sim_arr, meas)
    angle = params.get("angle_deg", 0.0)

    # --- Crop ---
    sim_crop  = _center_crop(sim_arr, crop_fraction)
    meas_crop = _center_crop(aligned_meas.displacement, crop_fraction)

    # Fill NaN with nearest valid value for display (measurement has holes
    # outside the Akro sensor footprint; sim is always solid).
    sim_crop  = _fill_nan_for_display(sim_crop)
    meas_crop = _fill_nan_for_display(meas_crop)

    # --- Downsample to ≤100×100 for smooth rendering ---
    step = max(1, max(sim_crop.shape) // 100)
    sim_ds  = sim_crop[::step, ::step]
    meas_ds = meas_crop[::step, ::step]

    ny, nx = sim_ds.shape
    x = np.arange(nx) * step
    y = np.arange(ny) * step

    # Residual (sim − meas)
    res = sim_ds - meas_ds

    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=sim_ds, x=x, y=y,
        name="Simulation",
        colorscale="Blues",
        opacity=0.75,
        showscale=False,
        showlegend=True,
    ))

    fig.add_trace(go.Surface(
        z=meas_ds, x=x, y=y,
        name="Measurement",
        colorscale="Reds",
        opacity=0.75,
        showscale=False,
        showlegend=True,
    ))

    fig.add_trace(go.Surface(
        z=res, x=x, y=y,
        name="Residual (Sim − Meas)",
        colorscale="RdBu",
        opacity=0.85,
        showscale=True,
        showlegend=True,
        visible="legendonly",   # hidden by default; toggle in legend
    ))

    source_label = str(row["source"])
    blur_label   = (f"{row['blur_type']} σ={float(row['sigma']):.1f}"
                    if row["blur_type"] != "none" else "no blur")
    dt_label     = (f"  ΔT={float(row['delta_t_c']):.0f}°C"
                    if source_label == "clt" else "")

    fig.update_layout(
        title=(
            f"{dat_file_stem}  |  {source_label}  {blur_label}{dt_label}  |  "
            f"DPI={dpi}  crop={crop_fraction:.0%}  align={angle:.2f}°"
        ),
        scene=dict(
            xaxis_title="X (px)",
            yaxis_title="Y (px)",
            zaxis_title="Displacement",
            aspectmode="data",
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    return fig.to_html(include_plotlyjs=True, full_html=True)


def _build_gradient_html(row: dict, stackup_hint: Path | None) -> str:
    """2D gradient vector field overlay: sim (blue) and measurement (red) arrows
    on a shared heatmap background.  Parallel arrows = good shape agreement,
    anti-parallel = consistent sign flip, perpendicular = no correlation."""
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.ndimage import gaussian_filter as gf
    from src.analysis.alignment import align
    from src.ingestion.akrometrix import load_akrometrix_dat
    from src.processing.denoise import denoise as _denoise
    from src.processing.inpainting import fill_missing as _fill_missing

    dat_file_stem = str(row.get("dat_file") or row["name"])
    dpi           = int(row["dpi"])
    material      = str(row["material"])
    fill_missing  = _parse_bool(row["fill_missing"])
    denoise_sigma = float(row["denoise_sigma"])
    crop_fraction = float(row["crop_fraction"])

    stored_akro = row.get("akro_folder", "")
    if stored_akro and Path(stored_akro).is_dir():
        candidates = list(Path(stored_akro).glob(f"{dat_file_stem}.dat"))
        dat_path = candidates[0] if candidates else _find_dat(dat_file_stem)
    else:
        dat_path = _find_dat(dat_file_stem)

    meas = load_akrometrix_dat(dat_path)
    if fill_missing:
        meas = _fill_missing(meas)
    if denoise_sigma > 0:
        meas = _denoise(meas, sigma=denoise_sigma)

    stackup      = _find_stackup(material, stackup_hint)
    npz_folder   = _find_npz_folder(dpi, material)
    density_maps = _load_density_maps(npz_folder, stackup)
    pixel_size_m = 25.4e-3 / dpi
    sim_arr      = _build_sim_array(row, density_maps, stackup, pixel_size_m)
    aligned_meas, _ = align(sim_arr, meas)

    sim_crop  = _fill_nan_for_display(_center_crop(sim_arr, crop_fraction))
    meas_crop = _fill_nan_for_display(_center_crop(aligned_meas.displacement, crop_fraction))

    # Downsample to ≤100×100 then smooth before gradients
    step = max(1, max(sim_crop.shape) // 100)
    sim_ds  = sim_crop[::step, ::step].astype(np.float64)
    meas_ds = meas_crop[::step, ::step].astype(np.float64)
    ny, nx  = sim_ds.shape
    x = np.arange(nx) * step
    y = np.arange(ny) * step

    smooth_sigma = 3.0
    sim_sm  = gf(sim_ds,  sigma=smooth_sigma)
    meas_sm = gf(meas_ds, sigma=smooth_sigma)

    # Compute gradients: np.gradient returns (dy, dx)
    gy_sim,  gx_sim  = np.gradient(sim_sm)
    gy_meas, gx_meas = np.gradient(meas_sm)

    # Arrow grid — aim for ~15 arrows per dimension
    q_step = max(1, max(ny, nx) // 15)
    xi = np.arange(0, nx, q_step)
    yi = np.arange(0, ny, q_step)
    XX, YY = np.meshgrid(x[xi], y[yi])

    def _quiver_traces(gx, gy, color, name):
        u = gx[np.ix_(yi, xi)]
        v = gy[np.ix_(yi, xi)]
        mag = np.sqrt(u**2 + v**2)

        # Only draw arrows where there is meaningful slope —
        # flat regions (noise) get masked so they don't show random directions.
        threshold = float(np.percentile(mag, 20))  # bottom 20% treated as flat
        active = mag > max(threshold, 1e-8)

        xx_a = XX.ravel()[active.ravel()]
        yy_a = YY.ravel()[active.ravel()]
        ua   = u.ravel()[active.ravel()]
        va   = v.ravel()[active.ravel()]
        mag_a = mag.ravel()[active.ravel()] + 1e-12

        scale = float(q_step * step * 0.7)
        un = ua / mag_a * scale
        vn = va / mag_a * scale

        if len(xx_a) == 0:
            return ()

        qfig = ff.create_quiver(
            xx_a, yy_a, un, vn,
            scale=1.0, arrow_scale=0.25,
            name=name, line=dict(color=color, width=1.5),
        )
        return qfig.data

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Simulation gradients", "Measurement gradients"],
        shared_yaxes=True,
    )

    # Left: sim heatmap + both gradient fields overlaid
    fig.add_trace(go.Heatmap(z=sim_ds, x=x, y=y, colorscale="Greys",
                             showscale=False, name="Sim surface"), row=1, col=1)
    for t in _quiver_traces(gx_sim, gy_sim, "royalblue", "Sim ∇"):
        fig.add_trace(t, row=1, col=1)
    for t in _quiver_traces(gx_meas, gy_meas, "tomato", "Meas ∇"):
        fig.add_trace(t, row=1, col=1)

    # Right: meas heatmap + both gradient fields overlaid
    fig.add_trace(go.Heatmap(z=meas_ds, x=x, y=y, colorscale="Greys",
                             showscale=False, name="Meas surface"), row=1, col=2)
    for t in _quiver_traces(gx_sim, gy_sim, "royalblue", "Sim ∇ "):
        fig.add_trace(t, row=1, col=2)
    for t in _quiver_traces(gx_meas, gy_meas, "tomato", "Meas ∇ "):
        fig.add_trace(t, row=1, col=2)

    source_label = str(row["source"])
    angle_str = (f"angle_mean={float(row.get('angle_mean_deg', float('nan'))):.1f}°"
                 if "angle_mean_deg" in row else "")

    fig.update_layout(
        title=f"{dat_file_stem}  |  {source_label}  DPI={dpi}  {angle_str}<br>"
              f"<b>Blue = Simulation gradient  ·  Red = Measurement gradient</b>",
        height=650,
        margin=dict(l=20, r=20, t=80, b=20),
    )
    fig.update_xaxes(scaleanchor="y", scaleratio=1)

    return fig.to_html(include_plotlyjs=True, full_html=True)


# ---------------------------------------------------------------------------
# Background loader
# ---------------------------------------------------------------------------

class _LoadWorker(QThread):
    ready = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, row: dict, stackup_hint: Path | None, mode: str = "surface"):
        super().__init__()
        self._row    = row
        self._hint   = stackup_hint
        self._mode   = mode

    def run(self) -> None:
        try:
            if self._mode == "gradient":
                html = _build_gradient_html(self._row, self._hint)
            else:
                html = _build_html(self._row, self._hint)
            self.ready.emit(html)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Dialog
# ---------------------------------------------------------------------------

class FitViewerDialog(QDialog):
    """Pop-up dialog showing overlaid simulation + measurement surfaces."""

    def __init__(self, row: dict, stackup_hint: Path | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(
            f"3D Fit — {row.get('name', '')}  |  "
            f"{row.get('source', '')}  DPI={row.get('dpi', '')}"
        )
        self.resize(1100, 750)
        self._row    = row
        self._hint   = stackup_hint
        self._worker: _LoadWorker | None = None
        self._build_ui()
        self._start_load()

    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        r = self._row
        blur = (f"{r.get('blur_type', 'none')} σ={float(r.get('sigma', 0)):.1f}"
                if r.get("blur_type", "none") != "none" else "no blur")
        info = (
            f"Material: {r.get('material', '—')}  |  "
            f"Source: {r.get('source', '—')}  |  "
            f"DPI: {r.get('dpi', '—')}  |  "
            f"Blur: {blur}  |  "
            f"ΔT: {float(r.get('delta_t_c', 0)):.1f}°C  |  "
            f"Crop: {float(r.get('crop_fraction', 1.0)):.0%}  |  "
            f"RMSE-Z: {float(r.get('rmse_z', float('nan'))):.4f} mm"
        )
        layout.addWidget(QLabel(info))

        self._status = QLabel("Loading surfaces…")
        layout.addWidget(self._status)

        if _HAS_WEBENGINE:
            self._view = QWebEngineView()
            layout.addWidget(self._view, stretch=1)
        else:
            layout.addWidget(QLabel(
                "PyQt6-WebEngine is not installed — cannot render 3D plot.\n"
                "Install with:  pip install PyQt6-WebEngine"
            ))
            self._view = None

        btn_row = QHBoxLayout()
        reload_btn = QPushButton("3D Surfaces")
        reload_btn.clicked.connect(self._start_load)
        grad_btn = QPushButton("Gradient Field")
        grad_btn.clicked.connect(self._start_gradient)
        close_btn  = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(reload_btn)
        btn_row.addWidget(grad_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------

    def _start_load(self) -> None:
        self._launch_worker("surface", "Loading surfaces…")

    def _start_gradient(self) -> None:
        self._launch_worker("gradient", "Computing gradient fields…")

    def _launch_worker(self, mode: str, status_msg: str) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._status.setText(status_msg)
        self._worker = _LoadWorker(self._row, self._hint, mode=mode)
        self._worker.ready.connect(self._on_ready)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_ready(self, html: str) -> None:
        self._status.setText("Ready — toggle traces in the legend.")
        if self._view:
            # setHtml has a ~2 MB limit; Plotly JS alone exceeds it.
            # Write to a temp file and load via URL instead.
            tmp = tempfile.NamedTemporaryFile(
                suffix=".html", delete=False, mode="w", encoding="utf-8"
            )
            tmp.write(html)
            tmp.close()
            self._tmp_html = tmp.name
            self._view.setUrl(QUrl.fromLocalFile(tmp.name))

    def _on_error(self, msg: str) -> None:
        self._status.setText(f"Error: {msg}")
        QMessageBox.critical(self, "Fit viewer error", msg)

    def closeEvent(self, event) -> None:
        tmp = getattr(self, "_tmp_html", None)
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        super().closeEvent(event)
