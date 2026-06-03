"""Biharmonic inpainting for Akrometrix measurement grids.

Fills NaN pixels by solving the biharmonic PDE (∇⁴u = 0), which is the
thin-plate bending equation — the same physics that governs PCB warpage.
Missing values are predicted by smooth continuation of the surrounding
surface curvature rather than simple interpolation.
"""

from __future__ import annotations

import numpy as np
from skimage.restoration import inpaint_biharmonic

from src.models import MeasurementData


def fill_missing(measurement: MeasurementData) -> MeasurementData:
    """Return a new MeasurementData with NaN pixels filled via biharmonic inpainting.

    Each spatially disconnected missing region is solved independently
    (``split_into_regions=True``), which keeps the sparse system small
    and avoids corner-region bleed across the board interior.

    If there are no missing pixels the original object is returned unchanged.
    """
    disp = measurement.displacement
    mask = ~np.isfinite(disp)

    if not mask.any():
        return measurement

    # Replace NaN with 0 before passing to inpaint (it ignores those cells,
    # but the array must be finite for the solver to initialise correctly).
    disp_clean = np.where(mask, 0.0, disp).astype(np.float64)

    filled = inpaint_biharmonic(
        disp_clean,
        mask,
        split_into_regions=True,
    ).astype(np.float32)

    return MeasurementData(
        source_file=measurement.source_file,
        displacement=filled,
        x_coords=measurement.x_coords.copy(),
        y_coords=measurement.y_coords.copy(),
    )


def missing_mask(measurement: MeasurementData) -> np.ndarray:
    """Return a boolean array: True where displacement is NaN."""
    return ~np.isfinite(measurement.displacement)
