"""Per-array comparison metrics: Kabsch/Umeyama 3-D alignment + statistics.

Ported from CopperBalancingFinal/lib/array_operations/comparison.py and
adapted for warpage maps (NaN-masked, zero-displacement is valid data).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_points(arr: NDArray) -> NDArray:
    """Return Nx3 array of (x, y, z) for all finite pixels."""
    ny, nx = arr.shape
    y, x = np.mgrid[0:ny, 0:nx]
    z = np.asarray(arr, dtype=float)
    mask = np.isfinite(z)
    return np.column_stack([x[mask].ravel(), y[mask].ravel(), z[mask].ravel()])


def _kabsch_umeyama(P: NDArray, Q: NDArray, with_scaling: bool = False):
    """Optimal rigid transform from P → Q via Kabsch/Umeyama SVD."""
    cp = P.mean(axis=0)
    cq = Q.mean(axis=0)
    Pc = P - cp
    Qc = Q - cq
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    if with_scaling:
        var_P = np.var(Pc, axis=0).sum()
        s = S.sum() / var_P if var_P > 0 else 1.0
    else:
        s = 1.0
    t = cq - s * (R @ cp)
    return R, t, s


def _fit_plane(points: NDArray) -> NDArray:
    X = points[:, :2]
    z = points[:, 2]
    A = np.column_stack([X, np.ones(len(X))])
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    return coeffs


def _detrend_plane(points: NDArray) -> NDArray:
    a, b, c = _fit_plane(points)
    d = points.copy()
    d[:, 2] -= a * points[:, 0] + b * points[:, 1] + c
    return d


def _downsample(P: NDArray, Q: NDArray, maxN: int, seed: int = 123):
    if P.shape[0] <= maxN:
        return P, Q
    rng = np.random.RandomState(seed)
    idx = rng.choice(P.shape[0], size=maxN, replace=False)
    return P[idx], Q[idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def align_and_compare(
    A: NDArray,
    B: NDArray,
    *,
    detrend: bool = True,
    with_scaling: bool = False,
    maxN_align: int = 20_000,
) -> dict:
    """Align A → B with a rigid (optionally scaled) SVD transform, then compute metrics.

    Both A and B must be 2-D arrays on the **same pixel grid** — i.e. B has
    already been spatially registered to A via ``alignment.align`` before
    calling this function.  The Kabsch step captures any residual
    misalignment and provides the rotation / translation / scale diagnostics.

    Parameters
    ----------
    A : 2-D ndarray — simulation (reference).
    B : 2-D ndarray — aligned measurement.
    detrend : remove best-fit plane before computing stats (default True).
    with_scaling : allow uniform scale in alignment (default False).
    maxN_align : cap on points used for the Kabsch fit (downsampled if needed).

    Returns
    -------
    dict with keys:
        scale, R00–R22, t_x, t_y, t_z
        rmse_3d, mae_3d, p95_3d, max_3d
        rmse_z,  mae_z,  p95_z,  max_z
        pearson_r, slope, intercept, r2
        n, detrended, with_scaling
    """
    P = _as_points(A)   # sim points
    Q = _as_points(B)   # measurement points

    # Intersect on (x, y) so we only compare pixels valid in both arrays
    kP = {(int(p[0]), int(p[1])) for p in P}
    kQ = {(int(q[0]), int(q[1])) for q in Q}
    common = sorted(kP & kQ)
    if not common:
        raise ValueError("No overlapping finite pixels between simulation and measurement.")

    az = {(int(p[0]), int(p[1])): p[2] for p in P}
    bz = {(int(q[0]), int(q[1])): q[2] for q in Q}
    Pfull = np.array([(x, y, az[(x, y)]) for x, y in common], dtype=float)
    Qfull = np.array([(x, y, bz[(x, y)]) for x, y in common], dtype=float)

    Pfit, Qfit = _downsample(Pfull, Qfull, maxN_align)
    R, t, s = _kabsch_umeyama(Pfit, Qfit, with_scaling=with_scaling)
    Paligned = s * (Pfull @ R.T) + t

    if detrend:
        Paligned = _detrend_plane(Paligned)
        Qd = _detrend_plane(Qfull)
    else:
        Qd = Qfull

    diff3d = Paligned - Qd
    dists  = np.linalg.norm(diff3d, axis=1)
    dz     = Paligned[:, 2] - Qd[:, 2]

    rmse_3d = float(np.sqrt(np.mean(dists ** 2)))
    mae_3d  = float(np.mean(np.abs(dists)))
    p95_3d  = float(np.percentile(dists, 95))
    max_3d  = float(np.max(dists))

    rmse_z = float(np.sqrt(np.mean(dz ** 2)))
    mae_z  = float(np.mean(np.abs(dz)))
    p95_z  = float(np.percentile(np.abs(dz), 95))
    max_z  = float(np.max(np.abs(dz)))

    zA, zB = Paligned[:, 2], Qd[:, 2]
    pearson_r = float(np.corrcoef(zA, zB)[0, 1]) if zA.size > 1 else float("nan")
    Areg = np.column_stack([zA, np.ones_like(zA)])
    a_coef, b_coef = np.linalg.lstsq(Areg, zB, rcond=None)[0]
    yhat  = a_coef * zA + b_coef
    ss_res = float(np.sum((zB - yhat) ** 2))
    ss_tot = float(np.sum((zB - zB.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Z metrics after z-scale matching via linear regression.
    # The Kabsch scale s ≈ 1 because pixel x,y coords (0–N) dwarf z (mm),
    # so Paligned z is effectively unscaled. Using regression residuals gives
    # rmse_z that reflects shape mismatch only, not amplitude difference.
    dz_scaled = zB - yhat
    rmse_z = float(np.sqrt(np.mean(dz_scaled ** 2)))
    mae_z  = float(np.mean(np.abs(dz_scaled)))
    p95_z  = float(np.percentile(np.abs(dz_scaled), 95))
    max_z  = float(np.max(np.abs(dz_scaled)))

    return {
        "scale":       float(s),
        "R00": float(R[0, 0]), "R01": float(R[0, 1]), "R02": float(R[0, 2]),
        "R10": float(R[1, 0]), "R11": float(R[1, 1]), "R12": float(R[1, 2]),
        "R20": float(R[2, 0]), "R21": float(R[2, 1]), "R22": float(R[2, 2]),
        "t_x": float(t[0]), "t_y": float(t[1]), "t_z": float(t[2]),
        "rmse_3d": rmse_3d, "mae_3d": mae_3d, "p95_3d": p95_3d, "max_3d": max_3d,
        "rmse_z":  rmse_z,  "mae_z":  mae_z,  "p95_z":  p95_z,  "max_z":  max_z,
        "pearson_r": pearson_r, "slope": float(a_coef),
        "intercept": float(b_coef), "r2": r2,
        "n": int(len(common)),
        "detrended":    bool(detrend),
        "with_scaling": bool(with_scaling),
    }
