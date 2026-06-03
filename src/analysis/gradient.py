"""Gradient computation and comparison for 2D displacement surfaces.

Ported from CopperBalancingFinal/lib/array_operations/gradient_analysis.py
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import RegularGridInterpolator

from src.models import GradientMetrics, MeasurementData, SimResult


def compute_gradients(
    Z: NDArray,
    dx: float = 1.0,
    dy: float = 1.0,
    method: str = "finite",
    window_size: int = 3,
) -> tuple[NDArray, NDArray]:
    """Return (gx, gy) gradient components for a 2-D surface.

    method="finite"  — central finite differences (fast).
    method="plane"   — least-squares plane fit in an N×N window (slower, smoother).
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("Z must be 2D")

    if method == "finite":
        dZ_dy, dZ_dx = np.gradient(Z, dy, dx)
        return dZ_dx, dZ_dy

    if method == "plane":
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 3")
        ny, nx = Z.shape
        half = window_size // 2
        gx = np.zeros_like(Z)
        gy = np.zeros_like(Z)
        wy = np.arange(-half, half + 1) * dy
        wx = np.arange(-half, half + 1) * dx
        WY, WX = np.meshgrid(wy, wx, indexing="ij")
        X_flat, Y_flat = WX.ravel(), WY.ravel()
        A_base = np.column_stack([X_flat, Y_flat, np.ones_like(X_flat)])
        for i in range(ny):
            for j in range(nx):
                r0, r1 = max(0, i - half), min(ny, i + half + 1)
                c0, c1 = max(0, j - half), min(nx, j + half + 1)
                patch = Z[r0:r1, c0:c1]
                pr, pc = patch.shape
                A = A_base.reshape(window_size, window_size, 3)[:pr, :pc].reshape(-1, 3)
                coeffs, *_ = np.linalg.lstsq(A, patch.ravel(), rcond=None)
                gx[i, j], gy[i, j] = coeffs[0], coeffs[1]
        return gx, gy

    raise ValueError("method must be 'finite' or 'plane'")


def compare_gradient_fields(
    gx_m: NDArray,
    gy_m: NDArray,
    gx_r: NDArray,
    gy_r: NDArray,
    eps: float = 1e-12,
) -> tuple[dict[str, float], NDArray, NDArray]:
    """Compare model vs reference gradient fields.

    Returns
    -------
    metrics : dict
        angle_mean_deg, angle_median_deg, angle_p95_deg,
        mag_ratio_mean, mag_ratio_median, mag_ratio_p05, mag_ratio_p95
    angle_diff : 2D ndarray
        Per-pixel angle difference (degrees) between model and reference.
    mag_ratio : 2D ndarray
        Per-pixel magnitude ratio |∇ref| / |∇model|.
    """
    g_m = np.stack([gx_m, gy_m], axis=-1)
    g_r = np.stack([gx_r, gy_r], axis=-1)

    mag_m = np.linalg.norm(g_m, axis=-1)
    mag_r = np.linalg.norm(g_r, axis=-1)

    dot = np.sum(g_m * g_r, axis=-1)
    cos_theta = np.clip(dot / (mag_m * mag_r + eps), -1.0, 1.0)
    angle_diff = np.degrees(np.arccos(cos_theta))
    mag_ratio = mag_r / (mag_m + eps)

    angle_flat = angle_diff.ravel()
    ratio_flat = mag_ratio.ravel()

    metrics: dict[str, float] = {
        "angle_mean_deg":    float(np.mean(angle_flat)),
        "angle_median_deg":  float(np.median(angle_flat)),
        "angle_p95_deg":     float(np.percentile(angle_flat, 95)),
        "mag_ratio_mean":    float(np.mean(ratio_flat)),
        "mag_ratio_median":  float(np.median(ratio_flat)),
        "mag_ratio_p05":     float(np.percentile(ratio_flat, 5)),
        "mag_ratio_p95":     float(np.percentile(ratio_flat, 95)),
    }
    return metrics, angle_diff, mag_ratio


def gradient_analysis(
    sim: SimResult,
    measurement: MeasurementData,
    method: str = "finite",
) -> tuple[GradientMetrics, NDArray, NDArray]:
    """High-level wrapper: compare gradients of sim vs measurement.

    Both arrays are expected to be on the same grid (measurement already
    aligned via ``alignment.align``).  If shapes differ, measurement is
    bilinearly resampled to match sim before computing gradients.

    Returns
    -------
    metrics : GradientMetrics
    angle_diff : 2D ndarray  — degrees, same shape as sim
    mag_ratio  : 2D ndarray  — unitless, same shape as sim
    """
    s = sim.displacement
    m = _resample(sim, measurement)

    valid = np.isfinite(s) & np.isfinite(m)
    s_clean = np.where(valid, s, 0.0)
    m_clean = np.where(valid, m, 0.0)

    gxs, gys = compute_gradients(s_clean, method=method)
    gxm, gym = compute_gradients(m_clean, method=method)

    raw, angle_diff, mag_ratio = compare_gradient_fields(gxs, gys, gxm, gym)

    angle_diff[~valid] = np.nan
    mag_ratio[~valid] = np.nan

    return (
        GradientMetrics(
            angle_mean_deg=raw["angle_mean_deg"],
            angle_median_deg=raw["angle_median_deg"],
            angle_p95_deg=raw["angle_p95_deg"],
            mag_ratio_mean=raw["mag_ratio_mean"],
            mag_ratio_median=raw["mag_ratio_median"],
            mag_ratio_p05=raw["mag_ratio_p05"],
            mag_ratio_p95=raw["mag_ratio_p95"],
        ),
        angle_diff,
        mag_ratio,
    )


def _resample(sim: SimResult, measurement: MeasurementData) -> NDArray:
    if measurement.displacement.shape == sim.displacement.shape:
        return measurement.displacement.astype(np.float64)
    interp = RegularGridInterpolator(
        (measurement.y_coords, measurement.x_coords),
        measurement.displacement,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
    gy, gx = np.meshgrid(sim.y_coords, sim.x_coords, indexing="ij")
    return interp(np.stack([gy, gx], axis=-1)).astype(np.float64)
