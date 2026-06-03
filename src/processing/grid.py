import numpy as np
from numpy.typing import NDArray


def make_grid(x_min: float, x_max: float, y_min: float, y_max: float, resolution_mm: float):
    """Create the common spatial grid all pipeline stages register to."""
    x = np.arange(x_min, x_max, resolution_mm)
    y = np.arange(y_min, y_max, resolution_mm)
    return x, y


def interpolate_to_grid(x_points: NDArray, y_points: NDArray, z_points: NDArray,
                         x_grid: NDArray, y_grid: NDArray) -> NDArray:
    """Interpolate a scattered point cloud onto a regular grid."""
    raise NotImplementedError
