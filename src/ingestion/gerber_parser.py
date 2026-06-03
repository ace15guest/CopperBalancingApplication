import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

_GERBV = Path(__file__).parents[2] / "assets" / "gerbv" / "gerbv.exe"


def gerber_to_png(
    gerb_file_path: str | Path,
    save_folder: str | Path,
    save_name: str,
    dpi: int = 300,
    outline_path: str | Path | None = None,
    anti_alias: bool = False,
    wait: bool = False,
) -> tuple[str, Path]:
    """
    Converts a Gerber file to PNG using gerbv.

    Parameters:
        gerb_file_path: Path to the input Gerber file.
        save_folder:    Folder where the output PNG will be saved.
        save_name:      Output filename without extension.
        dpi:            Render resolution. Default 300.
        outline_path:   Optional board outline Gerber to overlay.
        anti_alias:     Enable gerbv anti-aliasing (-a flag).
        wait:           Block until gerbv finishes. Use False for background rendering.

    Returns:
        Tuple of (command string, output PNG path).
    """
    gerb_file_path = Path(gerb_file_path).expanduser().resolve()
    outline_path = Path(outline_path).expanduser().resolve() if outline_path else None

    Path(save_folder).mkdir(parents=True, exist_ok=True)
    out_png_path = Path(save_folder) / f"{save_name}.png"

    if not _GERBV.exists():
        raise FileNotFoundError(f"gerbv executable not found at {_GERBV}")

    cmd = [str(_GERBV), "-x", "png", "-D", str(dpi)]
    if anti_alias:
        cmd.append("-a")
    cmd.extend(["-o", str(out_png_path), str(gerb_file_path)])
    if outline_path:
        cmd.append(str(outline_path))

    if wait:
        subprocess.run(cmd, check=True)
    else:
        subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return " ".join(cmd), out_png_path


def wait_for_file_stability(
    file_path: str | Path,
    check_interval: float = 0.1,
    stability_checks: int = 50,
    trial_times: int = 300,
) -> None:
    """
    Waits until a file exists and its size remains stable for a given number of checks.
    Used to confirm gerbv has finished writing a PNG before the caller reads it.

    Args:
        file_path:        Path to the file to monitor.
        check_interval:   Seconds between checks.
        stability_checks: Consecutive stable-size checks required before returning.
        trial_times:      Max iterations to wait for the file to appear.
    """
    file_path = Path(file_path)

    count = 0
    while not file_path.exists():
        time.sleep(check_interval)
        count += 1
        if count > trial_times:
            raise TimeoutError(f"File never appeared: {file_path}")

    previous_size = -1
    stable_count = 0
    while stable_count < stability_checks:
        current_size = os.path.getsize(file_path)
        if current_size == previous_size:
            stable_count += 1
        else:
            stable_count = 0
        previous_size = current_size
        time.sleep(check_interval)


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------

_GERBER_EXTENSIONS = {
    ".gbr", ".ger",
    ".gtl", ".gbl",   # top / bottom copper
    ".gts", ".gbs",   # top / bottom solder mask
    ".gto", ".gbo",   # top / bottom silkscreen
    ".gtp", ".gbp",   # top / bottom paste
    ".gm1", ".gko",   # board outline
}


@dataclass
class BatchResult:
    design_name: str
    dpi: int
    output_folder: Path
    outline_path: Path
    png_paths: list[Path] = field(default_factory=list)
    width_px: int = 0
    height_px: int = 0


def convert_folder(
    gerber_folder: str | Path,
    output_root: str | Path,
    design_name: str,
    dpi: int = 2000,
    anti_alias: bool = False,
) -> BatchResult:
    """
    Convert all Gerber files in a folder to PNG with a uniform canvas.

    Automatically generates a board outline from the union bounding box of all
    layers, saves it to the output folder, then renders each layer with that
    outline overlaid so every PNG is the same size.

    Output is saved to: output_root / {design_name}_{dpi}dpi /

    Args:
        gerber_folder:  Folder containing the Gerber files.
        output_root:    Parent folder for the output directory.
        design_name:    Used to name the output subfolder.
        dpi:            Render resolution passed to gerbv.
        anti_alias:     Enable gerbv anti-aliasing.

    Returns:
        BatchResult with output folder, outline path, PNG paths, and pixel dimensions.

    Raises:
        FileNotFoundError: No Gerber files found in gerber_folder.
        ValueError:        PNGs produced at different sizes (mismatched layers).
    """
    gerber_folder = Path(gerber_folder).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()

    out_folder = output_root / f"{design_name}_{dpi}dpi"
    out_folder.mkdir(parents=True, exist_ok=True)

    outline_gbr = gerber_folder / "outline.gbr"
    generate_outline_from_extrema(gerber_folder, outline_gbr)

    gerbers = sorted(
        f for f in gerber_folder.iterdir()
        if f.suffix.lower() in _GERBER_EXTENSIONS
        and f != outline_gbr
    )
    if not gerbers:
        raise FileNotFoundError(f"No Gerber files found in {gerber_folder}")

    png_paths: list[Path] = []
    for gerber in gerbers:
        _, png_path = gerber_to_png(
            gerb_file_path=gerber,
            save_folder=out_folder,
            save_name=gerber.stem,
            dpi=dpi,
            outline_path=outline_gbr,
            anti_alias=anti_alias,
            wait=True,
        )
        png_paths.append(png_path)

    # Verify all PNGs share the same pixel dimensions
    sizes: dict[str, tuple[int, int]] = {}
    for png in png_paths:
        with Image.open(png) as img:
            sizes[png.name] = img.size  # (width, height)

    unique_sizes = set(sizes.values())
    if len(unique_sizes) > 1:
        detail = "\n".join(f"  {name}: {w}×{h}px" for name, (w, h) in sorted(sizes.items()))
        raise ValueError(f"PNG size mismatch — layers rendered at different dimensions:\n{detail}")

    w, h = unique_sizes.pop()
    return BatchResult(
        design_name=design_name,
        dpi=dpi,
        output_folder=out_folder,
        outline_path=outline_gbr,
        png_paths=png_paths,
        width_px=w,
        height_px=h,
    )


# ---------------------------------------------------------------------------
# Outline generation
# ---------------------------------------------------------------------------

def _parse_gerber_extents(gerber_path: Path) -> tuple[float, float, float, float] | None:
    """
    Extract (x_min, y_min, x_max, y_max) in mm by parsing coordinate data directly.

    Strips Gerber parameter blocks and comments, then reads all X/Y coordinate
    words. Handles metric and imperial units and any valid FSLAX format spec.
    Returns None if no coordinates are found in the file.
    """
    try:
        content = gerber_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    # --- Format specification: %FSLAX{int}{dec}Y{int}{dec}*% ---
    fmt = re.search(r"%FSL[AI]X\d(\d)Y\d(\d)\*%", content)
    dec_x = int(fmt.group(1)) if fmt else 6
    dec_y = int(fmt.group(2)) if fmt else 6
    div_x = 10 ** dec_x
    div_y = 10 ** dec_y

    is_metric = "%MOMM*%" in content

    # Strip parameter blocks (%...%) and G04 comments so we only parse data words
    stripped = re.sub(r"%[^%]*%", "", content)
    stripped = re.sub(r"G04[^*]*\*", "", stripped)

    x_vals = [int(v) / div_x for v in re.findall(r"X(-?\d+)", stripped)]
    y_vals = [int(v) / div_y for v in re.findall(r"Y(-?\d+)", stripped)]

    if not x_vals or not y_vals:
        return None

    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    if not is_metric:
        x_min, y_min, x_max, y_max = (v * 25.4 for v in (x_min, y_min, x_max, y_max))

    return x_min, y_min, x_max, y_max


def generate_outline_from_extrema(
    gerber_folder: str | Path,
    output_path: str | Path,
    margin_mm: float = 1.0,
) -> Path:
    """
    Synthesize a board outline Gerber from the union bounding box of all layers.

    Parses coordinate data directly from each Gerber file — no rendering needed.
    Takes the union of all layer extents, expands by margin_mm on all sides, and
    writes a Gerber rectangle that encloses the full board area.

    The margin ensures the outline canvas is always larger than any individual
    layer (layer coordinates are feature centers; physical aperture width pushes
    the rendered edge slightly beyond the raw coordinate extrema).

    The resulting file can be passed as outline_path to convert_folder() so
    every layer is forced onto an identical canvas.

    Args:
        gerber_folder: Folder containing the Gerber files.
        output_path:   Destination path for the generated outline .gbr file.
        margin_mm:     Extra margin added to all four sides in mm. Default 1.0.

    Returns:
        Path to the generated outline Gerber file.

    Raises:
        FileNotFoundError: No Gerber files found in gerber_folder.
        ValueError:        No coordinate data found across all layers.
    """
    gerber_folder = Path(gerber_folder).expanduser().resolve()
    output_path = Path(output_path)

    gerbers = sorted(
        f for f in gerber_folder.iterdir()
        if f.suffix.lower() in _GERBER_EXTENSIONS
    )
    if not gerbers:
        raise FileNotFoundError(f"No Gerber files found in {gerber_folder}")

    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")

    for gerber in gerbers:
        extents = _parse_gerber_extents(gerber)
        if extents is None:
            continue
        gx_min, gy_min, gx_max, gy_max = extents
        x_min = min(x_min, gx_min)
        y_min = min(y_min, gy_min)
        x_max = max(x_max, gx_max)
        y_max = max(y_max, gy_max)

    if x_min == float("inf"):
        raise ValueError("No coordinate data found in any Gerber file")

    _write_outline_gerber(
        x_min - margin_mm, y_min - margin_mm,
        x_max + margin_mm, y_max + margin_mm,
        output_path,
    )
    return output_path


def _write_outline_gerber(
    x_min: float, y_min: float, x_max: float, y_max: float, output_path: Path
) -> None:
    """Write a minimal RS-274X Gerber rectangle from (x_min, y_min) to (x_max, y_max)."""
    # With FSLAX36Y36 + MOMM, coordinates are integers in units of 1e-6 mm
    def gu(mm: float) -> int:
        return round(mm * 1_000_000)

    x0, y0 = gu(x_min), gu(y_min)
    x1, y1 = gu(x_max), gu(y_max)

    content = (
        "%FSLAX36Y36*%\n"
        "%MOMM*%\n"
        "%ADD10C,0.10000*%\n"
        "G01*\n"
        "D10*\n"
        f"X{x0}Y{y0}D02*\n"
        f"X{x1}Y{y0}D01*\n"
        f"X{x1}Y{y1}D01*\n"
        f"X{x0}Y{y1}D01*\n"
        f"X{x0}Y{y0}D01*\n"
        "M02*\n"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
