"""Parameter sweep: DPI × blur × ΔT across three comparison tracks.

  IMBALANCE TRACK  (ΔT-independent)
    Gerbers → copper imbalance map (top−bottom weighted) → blur → compare vs Akro
    source = "imbalance"   blur_type = "none" | "gaussian" | "box"

  DENSITY TRACK    (ΔT-independent)
    Gerbers → total copper density map (unsigned sum) → blur → compare vs Akro
    source = "density"     blur_type = "none" | "gaussian" | "box"

  CLT TRACK        (sweeps ΔT)
    Gerbers → CLT solver → warpage map → compare vs Akro
    source = "clt"         blur_type = "none"   delta_t_c = swept value

One output row per (DPI, source, blur_type, sigma, radius, delta_t_c,
                    fill_missing, denoise_sigma, crop_fraction, dat_file).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import re
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_PROJECT_ROOT  = Path(__file__).parents[2]
_PNG_ROOT      = _PROJECT_ROOT / "assets" / "processed_pngs"
_NPZ_ROOT      = _PROJECT_ROOT / "assets" / "processed_npz"
_RESULTS_ROOT  = _PROJECT_ROOT / "assets" / "sweep_results"

# Columns that uniquely identify one sweep combination
_KEY_COLS = ("material", "dpi", "source", "blur_type", "sigma", "radius",
             "delta_t_c", "fill_missing", "denoise_sigma", "crop_fraction", "name", "dat_file")


def _rnd(v: float, decimals: int = 6) -> float:
    return round(float(v), decimals)


def _parse_bool(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() == "true"
    return bool(v)


def _row_key(row) -> tuple:
    return (str(row["material"]), int(row["dpi"]), str(row["source"]), str(row["blur_type"]),
            _rnd(row["sigma"]), _rnd(row["radius"]), _rnd(row["delta_t_c"]),
            _parse_bool(row["fill_missing"]), _rnd(row["denoise_sigma"]),
            _rnd(row["crop_fraction"]), str(row["name"]), str(row["dat_file"]))

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, uniform_filter

from src.analysis.alignment import align
from src.analysis.comparison import align_and_compare
from src.analysis.gradient import gradient_analysis
from src.ingestion.akrometrix import load_akrometrix_dat
from src.ingestion.gerber_parser import convert_folder
from src.models import CopperDensityMap, Stackup
from src.processing.denoise import denoise as _denoise
from src.processing.inpainting import fill_missing as _fill_missing
from src.processing.rasterizer import batch_png_to_npz, png_to_density_map
from src.simulation.clt_solver import solve_clt


# ---------------------------------------------------------------------------
# Output row
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SweepRow:
    # --- identity ---
    name:        str     # config entry name, e.g. "ACCL-EM890K_Q1"
    dat_file:    str     # Akrometrix .dat file stem, e.g. "ACCL-890K-01-Q1_Top_Global"
    location:    str     # Q-label parsed from dat_file, e.g. "Q1"
    side:        str     # "Top" or "Bottom" parsed from dat_file
    material:    str     # stackup material set name, e.g. "EM890K"
    dpi:         int
    source:      str     # "imbalance" | "density" | "clt"
    blur_type:   str     # "none" | "gaussian" | "box"
    sigma:       float   # Gaussian sigma (px); 0 for none/box
    radius:      float   # blur kernel radius (px); 0 for none
    delta_t_c:   float   # ΔT for CLT; 0.0 for imbalance/density tracks
    fill_missing:  bool
    denoise_sigma: float

    # --- Kabsch 3-D alignment diagnostics ---
    scale: float
    R00: float;  R01: float;  R02: float
    R10: float;  R11: float;  R12: float
    R20: float;  R21: float;  R22: float
    t_x: float;  t_y: float;  t_z: float

    # --- 3-D distance metrics ---
    rmse_3d: float;  mae_3d: float;  p95_3d: float;  max_3d: float

    # --- vertical (Z-only) metrics ---
    rmse_z: float;  mae_z: float;  p95_z: float;  max_z: float

    # --- correlation ---
    pearson_r: float;  slope: float;  intercept: float;  r2: float
    n: int;  detrended: bool;  with_scaling: bool

    crop_fraction: float  # fraction of each dimension kept (1.0 = full board)

    # --- gradient metrics ---
    angle_mean_deg:   float
    angle_median_deg: float
    angle_p95_deg:    float
    mag_ratio_mean:   float
    mag_ratio_median: float
    mag_ratio_p05:    float
    mag_ratio_p95:    float

    # --- source path (for fit viewer reload) ---
    akro_folder: str


# ---------------------------------------------------------------------------
# Filename parsing helpers
# ---------------------------------------------------------------------------

_Q_RE    = re.compile(r'Q(\d+)', re.IGNORECASE)
_SIDE_RE = re.compile(r'(Top|Bottom)', re.IGNORECASE)


def _parse_location(stem: str) -> str:
    m = _Q_RE.search(stem)
    return f"Q{m.group(1)}" if m else ""


def _parse_side(stem: str) -> str:
    m = _SIDE_RE.search(stem)
    return m.group(1).capitalize() if m else ""


# ---------------------------------------------------------------------------
# Blur helpers
# ---------------------------------------------------------------------------

def _apply_blur(arr: np.ndarray, blur_type: str, radius: float, sigma: float) -> np.ndarray:
    if blur_type == "gaussian":
        return gaussian_filter(arr, sigma=sigma, truncate=radius / sigma)
    if blur_type == "box":
        size = int(2 * radius + 1)
        return uniform_filter(arr, size=size)
    return arr.copy()


def _build_blur_ops(
    sigma_values: list[float],
    box_radius_values: list[float],
) -> list[tuple[str, float, float]]:
    """Return (blur_type, sigma, radius) tuples for imbalance and density tracks.

    blur_type: "none" | "gaussian" | "box".  sigma=0 adds a no-blur baseline.
    """
    ops: list[tuple[str, float, float]] = []
    has_none = False
    for sigma in sigma_values:
        if sigma <= 0:
            if not has_none:
                ops.append(("none", 0.0, 0.0))
                has_none = True
        else:
            radius = float(max(1, math.ceil(3.0 * sigma)))
            ops.append(("gaussian", sigma, radius))
    for box_r in box_radius_values:
        if box_r > 0:
            ops.append(("box", 0.0, float(box_r)))
    return ops


_OZ_THICKNESS_M = 1.4 * 25.4e-6   # 1 oz copper = 1.4 mil


def _compute_imbalance_map(
    density_maps: list[CopperDensityMap],
    stackup: Stackup,
) -> np.ndarray:
    """Weighted copper imbalance: Σ(density × oz × sign(z_mid)) over all copper layers.

    Positive = more copper above neutral axis → board bows upward on cooling.
    """
    from src.simulation.solver import LayerSpec, compute_z_coordinates, _classify_dielectric

    _MIL_TO_M = 25.4e-6

    # Build LayerSpecs to get z-coordinates
    rows = stackup.rows
    copper_indices = [i for i, r in enumerate(rows) if r.row_type == "copper"]
    outer_set = {copper_indices[0], copper_indices[-1]} if copper_indices else set()

    layers: list[LayerSpec] = []
    for i, row in enumerate(rows):
        if row.row_type == "copper":
            rc = "copper_outer" if i in outer_set else "copper_inner"
        else:
            rc = _classify_dielectric(row.material)
        layers.append(LayerSpec(
            row_class=rc,
            thickness_m=row.finish_thickness_mil * _MIL_TO_M,
            material=row.material,
            layer_number=row.layer_number,
        ))

    z_coords = compute_z_coordinates(layers)
    dm_by_label = {dm.layer_name: dm for dm in density_maps}
    grid_shape = density_maps[0].density.shape if density_maps else (200, 200)
    imbalance = np.zeros(grid_shape, dtype=np.float64)

    active = [l for l in layers if l.row_class != "ignore"]
    for layer, (z_bot, z_top) in zip(active, z_coords):
        if layer.row_class not in ("copper_outer", "copper_inner"):
            continue
        oz = layer.thickness_m / _OZ_THICKNESS_M
        z_mid = (z_bot + z_top) / 2.0
        sign = 1.0 if z_mid >= 0.0 else -1.0

        if layer.row_class == "copper_outer":
            density = np.ones(grid_shape, dtype=np.float64)
        else:
            label = f"L{layer.layer_number}"
            dm = dm_by_label.get(label)
            if dm is not None:
                density = dm.density.astype(np.float64)
                if density.shape != grid_shape:
                    from scipy.ndimage import zoom as _zoom
                    density = _zoom(density,
                                    (grid_shape[0] / density.shape[0],
                                     grid_shape[1] / density.shape[1]), order=1)
            else:
                density = np.full(grid_shape, 0.5)

        imbalance += sign * oz * density

    return imbalance


def _compute_density_map(
    density_maps: list[CopperDensityMap],
    stackup: Stackup,
) -> np.ndarray:
    """Total weighted copper density: Σ(density × oz) — unsigned, all layers."""
    from src.simulation.solver import LayerSpec, compute_z_coordinates, _classify_dielectric

    _MIL_TO_M = 25.4e-6
    rows = stackup.rows
    copper_indices = [i for i, r in enumerate(rows) if r.row_type == "copper"]
    outer_set = {copper_indices[0], copper_indices[-1]} if copper_indices else set()

    layers: list[LayerSpec] = []
    for i, row in enumerate(rows):
        if row.row_type == "copper":
            rc = "copper_outer" if i in outer_set else "copper_inner"
        else:
            rc = _classify_dielectric(row.material)
        layers.append(LayerSpec(
            row_class=rc,
            thickness_m=row.finish_thickness_mil * _MIL_TO_M,
            material=row.material,
            layer_number=row.layer_number,
        ))

    z_coords = compute_z_coordinates(layers)
    dm_by_label = {dm.layer_name: dm for dm in density_maps}
    grid_shape = density_maps[0].density.shape if density_maps else (200, 200)
    total = np.zeros(grid_shape, dtype=np.float64)

    active = [l for l in layers if l.row_class != "ignore"]
    for layer, (z_bot, z_top) in zip(active, z_coords):
        if layer.row_class not in ("copper_outer", "copper_inner"):
            continue
        oz = layer.thickness_m / _OZ_THICKNESS_M

        if layer.row_class == "copper_outer":
            density = np.ones(grid_shape, dtype=np.float64)
        else:
            label = f"L{layer.layer_number}"
            dm = dm_by_label.get(label)
            if dm is not None:
                density = dm.density.astype(np.float64)
                if density.shape != grid_shape:
                    from scipy.ndimage import zoom as _zoom
                    density = _zoom(density,
                                    (grid_shape[0] / density.shape[0],
                                     grid_shape[1] / density.shape[1]), order=1)
            else:
                density = np.full(grid_shape, 0.5)

        total += oz * density  # no sign — unsigned sum

    return total


def _center_crop(arr: np.ndarray, fraction: float) -> np.ndarray:
    """Return the central *fraction* of each dimension. 1.0 = no crop."""
    if fraction >= 1.0:
        return arr
    ny, nx = arr.shape
    dy = int(ny * (1.0 - fraction) / 2)
    dx = int(nx * (1.0 - fraction) / 2)
    return arr[dy: ny - dy, dx: nx - dx]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def count_sweep_combinations(
    n_dat_files: int,
    dpi_values: list[int],
    sigma_values: list[float],
    box_radius_values: list[float],
    delta_t_values: list[float],
    fill_missing_values: list[bool],
    denoise_sigma_values: list[float],
    crop_center_fractions: list[float],
) -> int:
    """Total number of output rows this sweep will produce (ignoring resume cache)."""
    blur_ops = _build_blur_ops(sigma_values, box_radius_values)
    preprocessing_combos = len(fill_missing_values) * len(denoise_sigma_values)
    return (
        n_dat_files
        * len(dpi_values)
        * (2 * len(blur_ops) + len(delta_t_values))
        * preprocessing_combos
        * len(crop_center_fractions)
    )


def run_sweep(
    akro_folder: Path,
    gerber_folder: Path,
    stackup: Stackup,
    delta_t_values: list[float],
    dpi_values: list[int],
    sigma_values: list[float],
    box_radius_values: list[float] | None = None,
    crop_center_fractions: list[float] | None = None,
    fill_missing_values: list[bool] | None = None,
    denoise_sigma_values: list[float] | None = None,
    board_name: str | None = None,
    cache_name: str | None = None,
    csv_path: Path | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Run CLT for every (DPI, ΔT, blur_op, preprocess) combination vs all .dat files.

    Parameters
    ----------
    akro_folder          : Directory of Akrometrix .dat files.
    gerber_folder        : Directory containing copper-layer Gerber files.
    stackup              : Board stackup (loaded from JSON).
    delta_t_values       : ΔT values (°C) to sweep.
    dpi_values           : Rasterisation DPI values to sweep.
    sigma_values         : Gaussian sigma values (px); 0 = no blur.
    box_radius_values    : Box blur half-widths (px).
    fill_missing_values  : Which fill-missing settings to sweep (e.g. [False, True]).
    denoise_sigma_values : Denoise Gaussian sigma values to sweep (0 = no denoise).
    progress_cb          : Optional callable(completed, total) for progress.
    """
    dat_files = sorted(akro_folder.glob("*.dat"))
    if not dat_files:
        raise ValueError(f"No .dat files found in {akro_folder}")

    fill_missing_values  = fill_missing_values  or [False]
    denoise_sigma_values = denoise_sigma_values or [0.0]

    # Load all measurements raw — preprocessing applied per combination below.
    raw_measurements = [load_akrometrix_dat(f) for f in dat_files]

    # Pre-compute preprocessing cache: (stem, fill, denoise_sigma) → MeasurementData.
    # Biharmonic fill is slow; doing it once up-front avoids repeating it per DPI/ΔT/blur.
    preprocessing_combos = [
        (fm, _rnd(ds))
        for fm in fill_missing_values
        for ds in denoise_sigma_values
    ]
    preproc_cache: dict[tuple, object] = {}
    logger.info(
        "Building preprocessing cache: %d .dat × %d combos ...",
        len(raw_measurements), len(preprocessing_combos),
    )
    for raw_m in raw_measurements:
        stem = Path(raw_m.source_file).stem
        for fm, ds in preprocessing_combos:
            m = raw_m
            if fm:
                m = _fill_missing(m)
            if ds > 0:
                m = _denoise(m, sigma=ds)
            preproc_cache[(stem, fm, ds)] = m
    logger.info("Preprocessing cache built (%d entries).", len(preproc_cache))

    # Blur ops shared by imbalance and density tracks (ΔT-independent)
    blur_ops = _build_blur_ops(sigma_values, box_radius_values or [])

    design   = board_name or gerber_folder.name
    material = stackup.material_set_name

    # --- CSV path ---
    if csv_path is None:
        csv_path = _RESULTS_ROOT / f"{design}_sweep.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # --- Load existing results for resume ---
    done_keys: set[tuple] = set()
    existing_df = pd.DataFrame()
    if csv_path.exists():
        try:
            existing_df = pd.read_csv(csv_path)
            if "material" not in existing_df.columns:
                existing_df.insert(3, "material", material)
                existing_df.to_csv(csv_path, index=False)
                logger.info("Migrated existing CSV: added 'material' column (filled with %r).", material)
        except pd.errors.ParserError:
            logger.warning("CSV has mixed schema — repairing %s", csv_path)
            existing_df = _repair_csv(csv_path, material)
        except Exception as exc:
            logger.warning("Could not read existing CSV (%s) — starting fresh", exc)
            existing_df = pd.DataFrame()
        try:
            for _, row in existing_df.iterrows():
                done_keys.add(_row_key(row))
            logger.info("Resume: loaded %d existing rows from %s", len(done_keys), csv_path)
        except Exception as exc:
            logger.warning("Could not build resume keys (%s) — starting fresh", exc)
            done_keys = set()
            existing_df = pd.DataFrame()

    # --- Split blur ops: CLT (no blur) sweeps all ΔT; Gaussian/box run once ---
    stems = [Path(m.source_file).stem for m in raw_measurements]
    crop_fractions = crop_center_fractions or [1.0]

    def _map_key(dpi, source, bt, sigma, radius, fm, ds, cf, stem):
        return (material, dpi, source, bt, _rnd(sigma), _rnd(radius), _rnd(0.0),
                bool(fm), ds, _rnd(cf), design, stem)

    def _clt_key(dpi, dt, fm, ds, cf, stem):
        return (material, dpi, "clt", "none", 0.0, 0.0, _rnd(dt),
                bool(fm), ds, _rnd(cf), design, stem)

    total = sum(
        1
        for dpi in dpi_values
        for source in ("imbalance", "density")
        for bt, sigma, radius in blur_ops
        for fm, ds in preprocessing_combos
        for cf in crop_fractions
        for stem in stems
        if _map_key(dpi, source, bt, sigma, radius, fm, ds, cf, stem) not in done_keys
    ) + sum(
        1
        for dpi in dpi_values
        for dt in delta_t_values
        for fm, ds in preprocessing_combos
        for cf in crop_fractions
        for stem in stems
        if _clt_key(dpi, dt, fm, ds, cf, stem) not in done_keys
    )
    total_possible = (
        len(dpi_values)
        * (2 * len(blur_ops) + len(delta_t_values))
        * len(preprocessing_combos) * len(crop_fractions) * len(dat_files)
    )
    skipped = total_possible - total
    completed = 0
    rows: list[SweepRow] = []

    logger.info(
        "Sweep starting: %d DPI × %d ΔT × %d blur ops × %d crops × %d .dat = %d remaining "
        "(%d skipped, results → %s)",
        len(dpi_values), len(delta_t_values), len(blur_ops), len(crop_fractions), len(dat_files),
        total, skipped, csv_path,
    )

    for dpi in dpi_values:
        density_maps = _rasterize_at_dpi(gerber_folder, stackup, dpi, design, cache_name or "")
        pixel_size_m = 25.4e-3 / dpi

        # ----------------------------------------------------------------
        # IMBALANCE TRACK — blur(top−bottom weighted map) vs Akro
        # ----------------------------------------------------------------
        imbalance_map = _compute_imbalance_map(density_maps, stackup)
        for bt, sigma, radius in blur_ops:
            w = _apply_blur(imbalance_map, bt, radius, sigma)
            for fm, ds in preprocessing_combos:
                for raw_m in raw_measurements:
                    stem = Path(raw_m.source_file).stem
                    meas = preproc_cache[(stem, fm, ds)]
                    if all(_map_key(dpi, "imbalance", bt, sigma, radius, fm, ds, cf, stem)
                           in done_keys for cf in crop_fractions):
                        continue
                    completed, rows = _run_comparison(
                        w, meas, stem, design, str(akro_folder), dpi, material, "imbalance", bt, sigma, radius, 0.0,
                        fm, ds, crop_fractions, done_keys, rows, csv_path,
                        completed, total, progress_cb,
                        key_fn=lambda cf, _bt=bt, _s=sigma, _r=radius, _fm=fm, _ds=ds:
                            _map_key(dpi, "imbalance", _bt, _s, _r, _fm, _ds, cf, stem),
                    )

        # ----------------------------------------------------------------
        # DENSITY TRACK — blur(total unsigned copper map) vs Akro
        # ----------------------------------------------------------------
        density_map = _compute_density_map(density_maps, stackup)
        for bt, sigma, radius in blur_ops:
            w = _apply_blur(density_map, bt, radius, sigma)
            for fm, ds in preprocessing_combos:
                for raw_m in raw_measurements:
                    stem = Path(raw_m.source_file).stem
                    meas = preproc_cache[(stem, fm, ds)]
                    if all(_map_key(dpi, "density", bt, sigma, radius, fm, ds, cf, stem)
                           in done_keys for cf in crop_fractions):
                        continue
                    completed, rows = _run_comparison(
                        w, meas, stem, design, str(akro_folder), dpi, material, "density", bt, sigma, radius, 0.0,
                        fm, ds, crop_fractions, done_keys, rows, csv_path,
                        completed, total, progress_cb,
                        key_fn=lambda cf, _bt=bt, _s=sigma, _r=radius, _fm=fm, _ds=ds:
                            _map_key(dpi, "density", _bt, _s, _r, _fm, _ds, cf, stem),
                    )

        # ----------------------------------------------------------------
        # CLT TRACK — CLT warpage vs Akro, sweeps ΔT
        # ----------------------------------------------------------------
        for dt in delta_t_values:
            if all(_clt_key(dpi, dt, fm, ds, cf, stem) in done_keys
                   for fm, ds in preprocessing_combos
                   for cf in crop_fractions for stem in stems):
                logger.info("DPI %d | ΔT=%.1f°C — CLT cached, skipping.", dpi, dt)
                continue
            logger.info("DPI %d | ΔT=%.1f°C — running CLT solve ...", dpi, dt)
            sim = solve_clt(stackup, density_maps, delta_temp_c=dt, pixel_size_m=pixel_size_m)
            w_clt = sim.displacement
            for fm, ds in preprocessing_combos:
                for raw_m in raw_measurements:
                    stem = Path(raw_m.source_file).stem
                    meas = preproc_cache[(stem, fm, ds)]
                    if all(_clt_key(dpi, dt, fm, ds, cf, stem) in done_keys
                           for cf in crop_fractions):
                        continue
                    completed, rows = _run_comparison(
                        w_clt, meas, stem, design, str(akro_folder), dpi, material, "clt", "none", 0.0, 0.0, dt,
                        fm, ds, crop_fractions, done_keys, rows, csv_path,
                        completed, total, progress_cb,
                        key_fn=lambda cf, _fm=fm, _ds=ds:
                            _clt_key(dpi, dt, _fm, _ds, cf, stem),
                    )

    logger.info("Sweep complete — %d new rows, %d cached rows", len(rows), len(existing_df))
    new_df = pd.DataFrame([dataclasses.asdict(r) for r in rows])
    if existing_df.empty:
        return new_df
    if new_df.empty:
        return existing_df.reset_index(drop=True)
    return pd.concat([existing_df, new_df], ignore_index=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _run_comparison(
    w: np.ndarray,
    meas,
    stem: str,
    board_name: str,
    akro_folder: str,
    dpi: int,
    material: str,
    source: str,
    blur_type: str,
    sigma: float,
    radius: float,
    delta_t_c: float,
    fm: bool,
    ds: float,
    crop_fractions: list[float],
    done_keys: set,
    rows: list,
    csv_path: Path,
    completed: int,
    total: int,
    progress_cb,
    key_fn,
) -> tuple[int, list]:
    """Align w to meas, compute metrics for each crop fraction, append rows."""
    try:
        aligned, _params = align(w, meas)
    except Exception:
        for cf in crop_fractions:
            if key_fn(cf) not in done_keys:
                completed += 1
                logger.warning("[%d/%d] align failed for %s — skipping", completed, total, stem)
                if progress_cb:
                    progress_cb(completed, total)
        return completed, rows

    try:
        grad_m, _, _ = gradient_analysis(_make_sim_result(w, aligned), aligned)
        g = dataclasses.asdict(grad_m)
    except Exception:
        g = _empty_grad()

    for cf in crop_fractions:
        key = key_fn(cf)
        if key in done_keys:
            continue

        w_crop = _center_crop(w, cf)
        a_crop = _center_crop(aligned.displacement, cf)

        try:
            cmp = align_and_compare(w_crop, a_crop, with_scaling=True)
        except Exception:
            cmp = _empty_cmp()

        rows.append(SweepRow(
            name=board_name,
            dat_file=stem,
            location=_parse_location(stem),
            side=_parse_side(stem),
            material=material,
            akro_folder=akro_folder,
            dpi=dpi,
            source=source,
            blur_type=blur_type,
            sigma=sigma,
            radius=radius,
            delta_t_c=delta_t_c,
            fill_missing=bool(fm),
            denoise_sigma=float(ds),
            scale=cmp["scale"],
            R00=cmp["R00"], R01=cmp["R01"], R02=cmp["R02"],
            R10=cmp["R10"], R11=cmp["R11"], R12=cmp["R12"],
            R20=cmp["R20"], R21=cmp["R21"], R22=cmp["R22"],
            t_x=cmp["t_x"], t_y=cmp["t_y"], t_z=cmp["t_z"],
            rmse_3d=cmp["rmse_3d"], mae_3d=cmp["mae_3d"],
            p95_3d=cmp["p95_3d"],   max_3d=cmp["max_3d"],
            rmse_z=cmp["rmse_z"],   mae_z=cmp["mae_z"],
            p95_z=cmp["p95_z"],     max_z=cmp["max_z"],
            pearson_r=cmp["pearson_r"], slope=cmp["slope"],
            intercept=cmp["intercept"], r2=cmp["r2"],
            n=cmp["n"], detrended=cmp["detrended"],
            with_scaling=cmp["with_scaling"],
            crop_fraction=cf,
            angle_mean_deg=g["angle_mean_deg"],
            angle_median_deg=g["angle_median_deg"],
            angle_p95_deg=g["angle_p95_deg"],
            mag_ratio_mean=g["mag_ratio_mean"],
            mag_ratio_median=g["mag_ratio_median"],
            mag_ratio_p05=g["mag_ratio_p05"],
            mag_ratio_p95=g["mag_ratio_p95"],
        ))

        done_keys.add(key)
        completed += 1

        row_df = pd.DataFrame([dataclasses.asdict(rows[-1])])
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        row_df.to_csv(csv_path, mode="a", header=write_header, index=False)

        logger.info(
            "[%d/%d] DPI=%d ΔT=%.1f type=%s fill=%s den=%.1f crop=%.2f dat=%s  rmse_z=%.4f",
            completed, total, dpi, delta_t_c, blur_type, fm, ds, cf, stem,
            rows[-1].rmse_z,
        )
        if progress_cb:
            progress_cb(completed, total)

    return completed, rows


def _rasterize_at_dpi(
    gerber_folder: Path,
    stackup: Stackup,
    dpi: int,
    design: str,
    cache_name: str = "",
) -> list[CopperDensityMap]:
    # Folder name: {cache_name}_{gerber_folder.name}_{dpi}dpi  e.g. Cu_Bal_TV_Q1_50dpi
    # Falls back to {design}_{dpi}dpi only if cache_name is not set.
    gerber_stem = gerber_folder.name
    if cache_name:
        folder_name = f"{cache_name}_{gerber_stem}_{dpi}dpi"
    else:
        folder_name = f"{design}_{dpi}dpi"

    npz_folder = _NPZ_ROOT / folder_name
    png_folder = _PNG_ROOT / folder_name

    # --- Priority 1: existing NPZs ---
    if npz_folder.exists():
        npzs = sorted(npz_folder.glob("*.npz"))
        if npzs:
            logger.info("DPI %d — reusing %d cached NPZs from %s", dpi, len(npzs), npz_folder)
            npz_by_stem = {p.stem: p for p in npzs}
            density_maps: list[CopperDensityMap] = []
            for row in stackup.rows:
                if row.row_type != "copper" or row.gerber_path is None:
                    continue
                stem = Path(row.gerber_path).stem
                npz_path = npz_by_stem.get(stem)
                if npz_path is None:
                    logger.warning("DPI %d — no NPZ for Gerber stem '%s'", dpi, stem)
                    continue
                data = np.load(npz_path)
                density_maps.append(CopperDensityMap(
                    layer_name=f"L{row.layer_number}",
                    density=data["density"].astype(np.float32),
                    x_coords=np.arange(data["density"].shape[1], dtype=np.float32),
                    y_coords=np.arange(data["density"].shape[0], dtype=np.float32),
                ))
            return density_maps

    # --- Priority 2: existing PNGs → convert to NPZ ---
    if png_folder.exists():
        batch_png_paths = sorted(png_folder.glob("*.png"))
        if batch_png_paths:
            logger.info("DPI %d — reusing %d cached PNGs from %s", dpi, len(batch_png_paths), png_folder)
            npz_folder.mkdir(parents=True, exist_ok=True)
            for png in batch_png_paths:
                np.savez_compressed(npz_folder / f"{png.stem}.npz",
                                    density=png_to_density_map(png, layer_name=png.stem).density)
            logger.info("DPI %d — wrote %d NPZs to %s", dpi, len(batch_png_paths), npz_folder)
            density_maps = []
            for row in stackup.rows:
                if row.row_type != "copper" or row.gerber_path is None:
                    continue
                stem = Path(row.gerber_path).stem
                npz_path = npz_folder / f"{stem}.npz"
                if not npz_path.exists():
                    continue
                data = np.load(npz_path)
                density_maps.append(CopperDensityMap(
                    layer_name=f"L{row.layer_number}",
                    density=data["density"].astype(np.float32),
                    x_coords=np.arange(data["density"].shape[1], dtype=np.float32),
                    y_coords=np.arange(data["density"].shape[0], dtype=np.float32),
                ))
            return density_maps

    # --- Priority 3: render from Gerbers → save PNG + NPZ ---
    logger.info("DPI %d — rasterising Gerbers → %s", dpi, png_folder)
    batch = convert_folder(gerber_folder, _PNG_ROOT, folder_name.rsplit(f"_{dpi}dpi", 1)[0], dpi=dpi)
    batch_png_paths = batch.png_paths
    logger.info("DPI %d — wrote %d PNGs to %s", dpi, len(batch_png_paths), png_folder)
    npz_folder.mkdir(parents=True, exist_ok=True)
    for png in batch_png_paths:
        np.savez_compressed(npz_folder / f"{png.stem}.npz",
                            density=png_to_density_map(png, layer_name=png.stem).density)
    logger.info("DPI %d — wrote %d NPZs to %s", dpi, len(batch_png_paths), npz_folder)
    density_maps = []
    for row in stackup.rows:
        if row.row_type != "copper" or row.gerber_path is None:
            continue
        stem = Path(row.gerber_path).stem
        png = next((p for p in batch_png_paths if p.stem == stem), None)
        if png is None:
            logger.warning("DPI %d — no PNG for Gerber stem '%s'", dpi, stem)
            continue
        density_maps.append(png_to_density_map(png, layer_name=f"L{row.layer_number}"))
    return density_maps


def _repair_csv(csv_path: Path, material: str) -> pd.DataFrame:
    """Repair a CSV that has mixed row widths from a partial schema migration.

    Old rows (no material column) have N fields; new rows have N+1.  We insert
    the material value into the short rows so the whole file is consistent.
    """
    import csv as _csv
    all_rows: list[list[str]] = []
    header: list[str] | None = None
    with csv_path.open(newline="", encoding="utf-8") as f:
        for i, fields in enumerate(_csv.reader(f)):
            if i == 0:
                header = list(fields)
            else:
                all_rows.append(list(fields))

    if not header:
        return pd.DataFrame()

    if "material" in header:
        # Header already correct; just drop rows with the wrong length.
        df = pd.DataFrame(
            [r for r in all_rows if len(r) == len(header)],
            columns=header,
        )
    else:
        n_old = len(header)
        new_header = header[:3] + ["material"] + header[3:]
        fixed: list[list[str]] = []
        for fields in all_rows:
            if len(fields) == n_old:
                fields = fields[:3] + [material] + fields[3:]
            if len(fields) == len(new_header):
                fixed.append(fields)
        df = pd.DataFrame(fixed, columns=new_header)

    df.to_csv(csv_path, index=False)
    logger.info("Repaired CSV: %d rows written to %s", len(df), csv_path)
    return df


def _make_sim_result(w: np.ndarray, aligned_meas):
    """Wrap a displacement array as a SimResult for gradient_analysis."""
    from src.models import SimResult
    ny, nx = w.shape
    return SimResult(
        mode="clt",
        displacement=w,
        x_coords=np.arange(nx, dtype=float),
        y_coords=np.arange(ny, dtype=float),
    )


def _empty_cmp() -> dict:
    nan = float("nan")
    return {
        "scale": nan,
        "R00": nan, "R01": nan, "R02": nan,
        "R10": nan, "R11": nan, "R12": nan,
        "R20": nan, "R21": nan, "R22": nan,
        "t_x": nan, "t_y": nan, "t_z": nan,
        "rmse_3d": nan, "mae_3d": nan, "p95_3d": nan, "max_3d": nan,
        "rmse_z": nan,  "mae_z": nan,  "p95_z": nan,  "max_z": nan,
        "pearson_r": nan, "slope": nan, "intercept": nan, "r2": nan,
        "n": 0, "detrended": True, "with_scaling": False,
    }


def _empty_grad() -> dict:
    nan = float("nan")
    return {
        "angle_mean_deg": nan, "angle_median_deg": nan, "angle_p95_deg": nan,
        "mag_ratio_mean": nan, "mag_ratio_median": nan,
        "mag_ratio_p05": nan,  "mag_ratio_p95": nan,
    }
