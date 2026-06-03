import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from PIL import Image
from src.models import CopperDensityMap
from src.ingestion.gerber_parser import BatchResult


def bitmap_to_array(bitmap_path: str | Path, inverted: bool = False) -> np.ndarray | None:
    """
    Convert a bitmap/PNG to a normalized numpy array.

    Args:
        bitmap_path: Path to the image file.
        inverted:    If True, invert pixel values (copper = dark instead of light).

    Returns:
        Grayscale numpy array with values 0–255, or None if loading fails.
    """
    try:
        with Image.open(bitmap_path) as img:
            gray = img.convert("L")
            array = np.array(gray, dtype=np.float32)

            max_val = np.max(array)
            if max_val > 0:
                array = array * 255.0 / max_val

            if inverted:
                array = 255 - array

            return array
    except Exception as e:
        print(f"Error converting bitmap at {bitmap_path}: {e}")
        return None


def png_to_density_map(png_path: str | Path, layer_name: str, inverted: bool = False) -> CopperDensityMap:
    """
    Load a Gerber-rendered PNG and return a normalised copper density map.
    Assumes white pixels = copper, black = substrate (pass inverted=True to flip).

    Raises:
        ValueError: If the image cannot be loaded.
    """
    array = bitmap_to_array(png_path, inverted=inverted)
    if array is None:
        raise ValueError(f"Failed to load PNG: {png_path}")

    density = array / 255.0  # 0.0 = no copper, 1.0 = full copper

    h, w = density.shape
    x_coords = np.arange(w, dtype=np.float32)
    y_coords = np.arange(h, dtype=np.float32)

    return CopperDensityMap(
        layer_name=layer_name,
        density=density,
        x_coords=x_coords,
        y_coords=y_coords,
    )


# ---------------------------------------------------------------------------
# NPZ export
# ---------------------------------------------------------------------------

@dataclass
class NpzBatchResult:
    design_name: str
    dpi: int
    output_folder: Path
    npz_paths: list[Path] = field(default_factory=list)


def png_to_npz(png_path: str | Path, npz_path: str | Path, inverted: bool = False) -> Path:
    """
    Convert a single Gerber-rendered PNG to a compressed NPZ file.

    The NPZ contains one array keyed 'density': a float32 array with values
    0.0 (no copper) to 1.0 (full copper), shape (height, width).

    Args:
        png_path:  Path to the source PNG.
        npz_path:  Destination path for the .npz file.
        inverted:  Pass True if copper renders dark instead of light.

    Returns:
        Path to the written NPZ file.
    """
    png_path = Path(png_path)
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    density_map = png_to_density_map(png_path, layer_name=png_path.stem, inverted=inverted)
    np.savez_compressed(npz_path, density=density_map.density)
    return npz_path


def batch_png_to_npz(
    batch: BatchResult,
    output_root: str | Path,
    inverted: bool = False,
) -> NpzBatchResult:
    """
    Convert all PNGs from a BatchResult to individual NPZ files.

    Output mirrors the PNG folder structure under output_root:
        output_root / {design_name}_{dpi}dpi / {layer_stem}.npz

    Args:
        batch:       Result from convert_folder().
        output_root: Parent folder for the NPZ output directory.
        inverted:    Pass True if copper renders dark instead of light.

    Returns:
        NpzBatchResult with output folder and list of NPZ paths.
    """
    output_root = Path(output_root).expanduser().resolve()
    out_folder = output_root / f"{batch.design_name}_{batch.dpi}dpi"
    out_folder.mkdir(parents=True, exist_ok=True)

    npz_paths: list[Path] = []
    for png_path in batch.png_paths:
        npz_path = png_to_npz(
            png_path=png_path,
            npz_path=out_folder / f"{png_path.stem}.npz",
            inverted=inverted,
        )
        npz_paths.append(npz_path)

    return NpzBatchResult(
        design_name=batch.design_name,
        dpi=batch.dpi,
        output_folder=out_folder,
        npz_paths=npz_paths,
    )
