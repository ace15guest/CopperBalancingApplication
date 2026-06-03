"""Noise reduction for Akrometrix displacement maps.

Three-step pipeline:
  1. Median pre-filter — large-window median removes systematic step-function
     offsets caused by soldermask/copper height discontinuities.  Median is
     robust to steps; Gaussian is not.  Set median_size=0 to skip.
  2. MAD outlier clip — flags pixels whose residual from the local median
     exceeds ``mad_threshold`` × MAD and replaces them with the local median.
     Targets sharp spike artefacts without affecting the low-frequency field.
  3. Gaussian smooth — suppresses remaining high-frequency noise.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from src.models import MeasurementData

_MEDIAN_WINDOW = 15   # px — local neighbourhood for MAD trend estimation


def denoise(
    measurement: MeasurementData,
    sigma: float = 2.0,
    mad_threshold: float = 5.0,
    median_size: int = 0,
) -> MeasurementData:
    """Return a denoised copy of *measurement*.

    Parameters
    ----------
    sigma:
        Gaussian smoothing radius in pixels.  0 disables smoothing.
    mad_threshold:
        Outliers further than this many MADs from the local median are
        replaced with the local median before Gaussian smoothing.
        0 disables outlier clipping.
    median_size:
        Window size (pixels) for the large-median pre-filter applied before
        MAD clipping.  Use odd values; 0 disables.  Recommended for
        soldermask/copper step-function artefacts on unpopulated boards —
        a window larger than the largest pad removes the offset entirely
        rather than just blurring it.
    """
    z = measurement.displacement.copy().astype(np.float64)
    valid = np.isfinite(z)

    if not valid.any():
        return measurement

    nan_fill = float(np.nanmedian(z[valid]))

    # --- Step 1: large median pre-filter (soldermask / Cu step removal) ---
    if median_size > 1:
        z_filled = np.where(valid, z, nan_fill)
        z_med = median_filter(z_filled, size=median_size)
        # Restore NaN border — only replace valid pixels
        z = np.where(valid, z_med, z)

    # --- Step 2: MAD outlier clip ---
    if mad_threshold > 0:
        z_filled = np.where(valid, z, nan_fill)
        win = max(3, min(_MEDIAN_WINDOW, z.shape[0] // 8, z.shape[1] // 8))
        local_med = median_filter(z_filled, size=win)
        residual = z - local_med
        finite_res = residual[valid]
        mad = float(np.median(np.abs(finite_res - np.median(finite_res))))
        if mad > 0:
            outlier = valid & (np.abs(residual) > mad_threshold * mad)
            z[outlier] = local_med[outlier]

    # --- Step 3: Gaussian smooth ---
    if sigma > 0:
        nan_mask = ~valid
        z_filled = np.where(nan_mask, 0.0, z)
        weight   = np.where(nan_mask, 0.0, 1.0)
        smooth_vals    = gaussian_filter(z_filled, sigma=sigma)
        smooth_weights = gaussian_filter(weight,   sigma=sigma)
        with np.errstate(invalid="ignore"):
            z = np.where(smooth_weights > 0, smooth_vals / smooth_weights, z)

    return MeasurementData(
        source_file=measurement.source_file,
        displacement=z.astype(np.float32),
        x_coords=measurement.x_coords.copy(),
        y_coords=measurement.y_coords.copy(),
    )
