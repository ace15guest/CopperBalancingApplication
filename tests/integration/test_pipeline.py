"""
Quick pipeline test — run from the project root:
    .venv/Scripts/python -m pytest tests/integration/test_pipeline.py -s
"""
from pathlib import Path
from src.ingestion.gerber_parser import convert_folder, generate_outline_from_extrema
from src.processing.rasterizer import batch_png_to_npz

GERBER_FOLDER = Path(r"C:\Users\Asa Guest\Documents\Projects\CopperBalancingApplication\assets\gerbers\Q1")
OUTPUT_ROOT   = Path("assets/processed_pngs")
NPZ_ROOT      = Path("assets/processed_npz")
DESIGN_NAME   = "Cu_Bal_TV_Q1"
DPI           = 50


def test_convert_folder():
    result = convert_folder(
        gerber_folder=GERBER_FOLDER,
        output_root=OUTPUT_ROOT,
        design_name=DESIGN_NAME,
        dpi=DPI,
    )
    print(f"\nOutput folder : {result.output_folder}")
    print(f"Outline       : {result.outline_path}")
    print(f"Dimensions    : {result.width_px} × {result.height_px} px")
    print(f"Layers converted ({len(result.png_paths)}):")
    for p in result.png_paths:
        print(f"  {p.name}")

    assert result.output_folder.exists()
    assert result.outline_path.exists()
    assert len(result.png_paths) > 0
    assert result.width_px > 0
    assert result.height_px > 0


def test_batch_png_to_npz():
    batch = convert_folder(
        gerber_folder=GERBER_FOLDER,
        output_root=OUTPUT_ROOT,
        design_name=DESIGN_NAME,
        dpi=DPI,
    )
    result = batch_png_to_npz(batch=batch, output_root=NPZ_ROOT)

    print(f"\nNPZ folder : {result.output_folder}")
    print(f"Layers saved ({len(result.npz_paths)}):")
    for p in result.npz_paths:
        print(f"  {p.name}")

    assert result.output_folder.exists()
    assert len(result.npz_paths) == len(batch.png_paths)
    for npz in result.npz_paths:
        assert npz.exists()
        import numpy as np
        data = np.load(npz)
        assert "density" in data
        assert data["density"].ndim == 2
        assert data["density"].min() >= 0.0
        assert data["density"].max() <= 1.0


def test_generate_outline_from_extrema():
    outline_path = GERBER_FOLDER / "generated_outline.gbr"
    result = generate_outline_from_extrema(
        gerber_folder=GERBER_FOLDER,
        output_path=outline_path,
    )
    print(f"\nOutline written to: {result}")
    print(f"Contents:\n{result.read_text()}")

    assert result.exists()
    assert result.suffix == ".gbr"
    assert "MOMM" in result.read_text()
    assert result.stat().st_size > 0
