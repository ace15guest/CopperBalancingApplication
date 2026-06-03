from pathlib import Path

import numpy as np

from src.models import MeasurementData

_SENTINEL = 9999.0


def load_akrometrix_dat(file_path: Path) -> MeasurementData:
    """Parse a tab-separated Akrometrix .dat displacement grid.

    Values in the file are in microns; 9999.0 marks out-of-board pixels.
    Returned displacement is in mm. Coordinates are pixel indices — call
    alignment.align() to register onto a Gerber reference with mm coords.
    """
    path = Path(file_path)
    rows: list[list[float]] = []
    with path.open("rb") as f:
        for raw_line in f:
            line = raw_line.decode("ascii", errors="replace").rstrip()
            if not line:
                continue
            vals: list[float] = []
            for tok in line.split("\t"):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    vals.append(float(tok))
                except ValueError:
                    vals.append(np.nan)
            if vals:
                rows.append(vals)

    if not rows:
        raise ValueError(f"No data found in {path}")

    max_cols = max(len(r) for r in rows)
    grid = np.full((len(rows), max_cols), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        grid[i, : len(row)] = row

    grid[grid == _SENTINEL] = np.nan
    grid /= 1000.0  # μm → mm

    n_rows, n_cols = grid.shape
    return MeasurementData(
        source_file=str(path),
        displacement=grid,
        x_coords=np.arange(n_cols, dtype=float),
        y_coords=np.arange(n_rows, dtype=float),
    )
