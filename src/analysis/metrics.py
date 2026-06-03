from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from src.models import ComparisonMetrics, MeasurementData, SimResult


def compute_metrics(sim: SimResult, measurement: MeasurementData) -> ComparisonMetrics:
    """Compute the full suite of comparison metrics.

    If the grids differ in shape, the measurement is bilinearly resampled
    onto the simulation grid before any calculation.
    Both arrays must share the same spatial extent (i.e. measurement has
    already been aligned via ``alignment.align``).
    """
    s = sim.displacement
    m = _resample_to_sim(sim, measurement)

    valid = np.isfinite(s) & np.isfinite(m)
    if valid.sum() < 4:
        raise ValueError("Too few valid overlapping pixels to compute metrics.")

    sv = s[valid]
    mv = m[valid]

    # RMS error
    rms = float(np.sqrt(np.mean((sv - mv) ** 2)))

    # R² (treats measurement as "true" signal)
    ss_res = float(np.sum((sv - mv) ** 2))
    ss_tot = float(np.sum((mv - mv.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-12)

    # Pearson correlation
    pearson = float(np.corrcoef(sv, mv)[0, 1])

    # Gradient correlation — correlate flattened gradient magnitudes
    gs_r, gs_c = np.gradient(np.where(np.isfinite(s), s, 0.0))
    gm_r, gm_c = np.gradient(np.where(np.isfinite(m), m, 0.0))
    gsv = np.hypot(gs_r[valid], gs_c[valid])
    gmv = np.hypot(gm_r[valid], gm_c[valid])
    if gsv.std() > 1e-12 and gmv.std() > 1e-12:
        grad_corr = float(np.corrcoef(gsv, gmv)[0, 1])
    else:
        grad_corr = 0.0

    # Hotspot overlap — IoU of top-10 % pixels by displacement magnitude
    thresh_s = float(np.percentile(sv, 90))
    thresh_m = float(np.percentile(mv, 90))
    hot_s = sv >= thresh_s
    hot_m = mv >= thresh_m
    union = int(np.sum(hot_s | hot_m))
    hotspot = float(np.sum(hot_s & hot_m)) / (union + 1e-12)

    # IPC-style bow ratio: peak-to-valley over board diagonal (both in mm)
    h, w = s.shape
    diag_px = float(np.sqrt(h ** 2 + w ** 2))
    ipc_bow = float(sv.max() - sv.min()) / (diag_px + 1e-12)

    return ComparisonMetrics(
        rms_error=rms,
        r_squared=r2,
        pearson=pearson,
        gradient_correlation=grad_corr,
        hotspot_overlap=hotspot,
        ipc_bow_ratio=ipc_bow,
    )


def difference_map(sim: SimResult, measurement: MeasurementData) -> np.ndarray:
    """Return element-wise (sim − measurement) on the sim grid, NaN where either is missing."""
    m = _resample_to_sim(sim, measurement)
    diff = sim.displacement - m
    diff[~(np.isfinite(sim.displacement) & np.isfinite(m))] = np.nan
    return diff


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resample_to_sim(sim: SimResult, measurement: MeasurementData) -> np.ndarray:
    """Bilinearly resample measurement onto sim's grid; identity if shapes match."""
    if measurement.displacement.shape == sim.displacement.shape:
        return measurement.displacement.astype(np.float32)

    interp = RegularGridInterpolator(
        (measurement.y_coords, measurement.x_coords),
        measurement.displacement,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    gy, gx = np.meshgrid(sim.y_coords, sim.x_coords, indexing="ij")
    return interp(np.stack([gy, gx], axis=-1)).astype(np.float32)
