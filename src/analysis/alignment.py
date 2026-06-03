"""Spatial registration of a DAT measurement onto a reference grid.

Algorithm (rotation + resample):
  1. Apply optional flip (x / y).
  2. Search for a small rotation angle (±rotation_limit_deg) that maximises
     mask overlap with the reference footprint.
  3. Apply the rotation in-place (reshape=False, same canvas size).
  4. Resample to the reference grid shape.

Akrometrix measurement points already fall within the board's Gerber
footprint, so aggressive crop / anisotropic scale / phase-shift transforms
are not applied — they distort the physical correspondence.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, rotate as nd_rotate, zoom
from scipy.optimize import minimize_scalar
from skimage.transform import resize as sk_resize

from src.models import MeasurementData, SimResult  # noqa: F401 (SimResult kept for API compat)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_rotation(
    crop: NDArray,
    reference: NDArray,
    coarse_steps: int = 21,
    search_limit: float = 5.0,
) -> float:
    """Find the CCW rotation (degrees) that maximises crop-mask / reference-mask overlap.

    Uses a downsampled coarse grid search followed by bounded scalar refinement.
    search_limit constrains the search to ±search_limit degrees.
    """
    MAX_DIM = 128

    def _ds(arr: NDArray) -> NDArray:
        r, c = arr.shape
        factor = min(MAX_DIM / r, MAX_DIM / c, 1.0)
        return zoom(arr, factor, order=1) if factor < 1.0 else arr

    ref_mask = _ds(np.isfinite(reference).astype(np.float32))
    ref_sm   = gaussian_filter(ref_mask, sigma=3.0)
    ref_sm  /= ref_sm.max() + 1e-9

    crop_mask = _ds(np.isfinite(crop).astype(np.float32))
    target_shape = ref_sm.shape

    def _score(angle: float) -> float:
        rot = nd_rotate(crop_mask, angle, reshape=False, order=1, cval=0.0)
        zy  = target_shape[0] / rot.shape[0]
        zx  = target_shape[1] / rot.shape[1]
        rs  = zoom(rot, (zy, zx), order=1)
        sm  = gaussian_filter(rs, sigma=3.0)
        sm /= sm.max() + 1e-9
        return -float(np.dot(ref_sm.ravel(), sm.ravel()))

    angles = np.linspace(-search_limit, search_limit, coarse_steps, endpoint=False)
    losses = np.array([_score(a) for a in angles])
    best   = float(angles[np.argmin(losses)])
    step   = 2.0 * search_limit / coarse_steps

    result = minimize_scalar(_score, bounds=(best - step, best + step), method="bounded")
    return float(result.x)


def _apply_rotation(disp: NDArray, angle_deg: float) -> tuple[NDArray, NDArray]:
    """Rotate *disp* CCW by *angle_deg* (reshape=False). Returns (rotated, valid_mask)."""
    valid = np.isfinite(disp).astype(np.float32)
    data  = np.where(valid.astype(bool), disp, 0.0).astype(np.float32)
    rot_data  = nd_rotate(data,  angle_deg, reshape=False, order=1,  cval=0.0)
    rot_valid = nd_rotate(valid, angle_deg, reshape=False, order=1,  cval=0.0)
    mask = rot_valid >= 0.5
    rot_data[~mask] = np.nan
    return rot_data, mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def align(
    reference: NDArray,
    measurement: MeasurementData,
    flip_x: bool = False,
    flip_y: bool = False,
    rotation_limit_deg: float = 5.0,
) -> tuple[MeasurementData, dict]:
    """Align *measurement* onto *reference* via rotation correction + resample.

    Akrometrix points already lie within the board's Gerber footprint, so
    only a small rotation correction and a resolution resample are applied.
    No bounding-box crop, no anisotropic scale, no phase-shift translation.

    Parameters
    ----------
    reference:
        2-D reference array (CLT displacement or density map).
        Its shape defines the output grid.
    measurement:
        Raw (or fill_missing-processed) MeasurementData.
    flip_x / flip_y:
        Mirror the DAT array before alignment.
    rotation_limit_deg:
        Maximum rotation search range (±degrees). Default 5°.

    Returns
    -------
    aligned_measurement:
        MeasurementData resampled onto ``reference.shape``.
    params:
        Dict with keys ``angle_deg``, ``flip_x``, ``flip_y``,
        ``scale_x``, ``scale_y``.
    """
    disp = measurement.displacement.copy()
    if flip_x:
        disp = np.fliplr(disp)
    if flip_y:
        disp = np.flipud(disp)

    ref_shape = reference.shape

    if not np.any(np.isfinite(reference)):
        raise ValueError("Reference array has no finite data.")
    if not np.any(np.isfinite(disp)):
        raise ValueError("Measurement array has no finite data.")

    # --- Step 1: small rotation correction ---
    angle_deg = _find_rotation(disp, reference, search_limit=rotation_limit_deg)
    if abs(angle_deg) > 0.05:
        disp, _ = _apply_rotation(disp, angle_deg)
    else:
        angle_deg = 0.0

    # --- Step 2: resample to reference grid shape ---
    valid = np.isfinite(disp)
    disp_clean = np.where(valid, disp, 0.0)
    disp_rs = sk_resize(
        disp_clean, ref_shape, order=1,
        anti_aliasing=True, preserve_range=True,
    ).astype(np.float32)
    mask_rs = sk_resize(
        valid.astype(np.float32), ref_shape, order=0,
        anti_aliasing=False, preserve_range=True,
    )
    disp_rs[mask_rs < 0.5] = np.nan

    n_rows, n_cols = ref_shape
    params = {
        "angle_deg": float(angle_deg),
        "flip_x":    bool(flip_x),
        "flip_y":    bool(flip_y),
        "scale_x":   ref_shape[1] / disp.shape[1],
        "scale_y":   ref_shape[0] / disp.shape[0],
    }
    return (
        MeasurementData(
            source_file=measurement.source_file,
            displacement=disp_rs,
            x_coords=np.arange(n_cols, dtype=float),
            y_coords=np.arange(n_rows, dtype=float),
        ),
        params,
    )


def apply_alignment(
    arr: NDArray,
    params: dict,
    out_shape: tuple[int, int],
    order: int = 1,
    cval: float = 0.0,
) -> NDArray:
    """Re-apply a previously computed alignment to another array."""
    a = np.asarray(arr, dtype=np.float32)
    p = {"angle_deg": 0.0, "flip_x": False, "flip_y": False, **params}

    if p["flip_x"]:
        a = np.fliplr(a)
    if p["flip_y"]:
        a = np.flipud(a)

    angle = p["angle_deg"]
    if abs(angle) > 0.05:
        a = nd_rotate(a, angle, reshape=False, order=order, cval=cval)

    return sk_resize(
        a, out_shape, order=order,
        anti_aliasing=(order > 0), preserve_range=True,
    ).astype(np.float32)
