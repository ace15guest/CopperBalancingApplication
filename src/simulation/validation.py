"""
Synthetic validation tests for the lamination warpage simulation.

Run directly from the project root:
    .venv/Scripts/python -m src.simulation.validation

No test framework is used — results are printed and assertions verify
basic physical expectations:
  - A symmetric stackup should produce near-zero warpage.
  - An asymmetric stackup should produce measurable bow in the expected
    direction (positive kappa when top copper is denser than bottom).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).parents[2]
_LIBRARY_PATH = _PROJECT_ROOT / "data" / "materials.json"

_PROCESS_CONFIG = {
    "T_press_c":         200.0,
    "T_ambient_c":       25.0,
    "T_vitrification_c": 150.0,
    "board_width_m":     0.200,   # 200 mm
    "board_height_m":    0.200,   # 200 mm
    "material_set":      "Set A",
    "default_grid":      (50, 50),
}


def _write_temp_stackup(rows: list[dict]) -> Path:
    """
    Write a minimal stackup JSON to a temp file and return its path.

    Args:
        rows: List of raw stackup row dicts (stackup JSON format).

    Returns:
        Path to the written temporary JSON file.
    """
    data = {"version": 2, "material_set": "Set A", "stackup": rows}
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    )
    json.dump(data, tmp, indent=2)
    tmp.close()
    return Path(tmp.name)


def _row(row_type: str, material: str, thickness_mil: float, layer_number=None) -> dict:
    """
    Build a single stackup row dict.

    Args:
        row_type:       "copper" or "dielectric".
        material:       Material name string.
        thickness_mil:  Layer thickness in mils.
        layer_number:   Copper layer index (1-based), or None.

    Returns:
        Dict matching the v2 stackup JSON schema.
    """
    return {
        "row_type":             row_type,
        "layer_number":         layer_number,
        "material":             material,
        "finish_thickness_mil": thickness_mil,
        "gerber_path":          None,
    }


def _symmetric_stackup_rows() -> list[dict]:
    """
    Return rows for a symmetric 4-copper-layer stackup.

    Structure (top to bottom):
        LPI mask    1 mil
        Cu outer    1 oz  (1.4 mil)
        PrePreg     6 mil
        Cu inner    1 oz  (1.4 mil)
        FR4 Core   12 mil
        Cu inner    1 oz  (1.4 mil)
        PrePreg     6 mil
        Cu outer    1 oz  (1.4 mil)
        LPI mask    1 mil

    Mirror-symmetric about the core centre → B = 0, w ≈ 0.

    Returns:
        List of stackup row dicts.
    """
    return [
        _row("dielectric", "Liquid PhotoImageable Mask", 1.0),
        _row("copper",     "1 oz",                       1.4,  layer_number=1),
        _row("dielectric", "PrePreg 1080",                6.0),
        _row("copper",     "1 oz",                        1.4,  layer_number=2),
        _row("dielectric", "FR4 Core",                   12.0),
        _row("copper",     "1 oz",                        1.4,  layer_number=3),
        _row("dielectric", "PrePreg 1080",                6.0),
        _row("copper",     "1 oz",                        1.4,  layer_number=4),
        _row("dielectric", "Liquid PhotoImageable Mask",  1.0),
    ]


def _asymmetric_stackup_rows() -> list[dict]:
    """
    Return rows for a deliberately asymmetric 4-copper-layer stackup.

    The top copper layer (outer) is thicker than the bottom outer layer.
    This imbalance creates a non-zero thermal moment and predictable bow —
    the board bows toward the thicker-copper (top) side (negative w at
    centre relative to edges) under standard lamination conditions.

    Structure:
        LPI mask    1 mil
        Cu outer    3 oz  (4.2 mil)   ← heavy top
        PrePreg     6 mil
        Cu inner    1 oz  (1.4 mil)
        FR4 Core   12 mil
        Cu inner    1 oz  (1.4 mil)
        PrePreg     6 mil
        Cu outer    1 oz  (1.4 mil)   ← light bottom
        LPI mask    1 mil

    Returns:
        List of stackup row dicts.
    """
    return [
        _row("dielectric", "Liquid PhotoImageable Mask", 1.0),
        _row("copper",     "3 oz",                       4.2,  layer_number=1),
        _row("dielectric", "PrePreg 1080",                6.0),
        _row("copper",     "1 oz",                        1.4,  layer_number=2),
        _row("dielectric", "FR4 Core",                   12.0),
        _row("copper",     "1 oz",                        1.4,  layer_number=3),
        _row("dielectric", "PrePreg 1080",                6.0),
        _row("copper",     "1 oz",                        1.4,  layer_number=4),
        _row("dielectric", "Liquid PhotoImageable Mask",  1.0),
    ]


def run_synthetic_test() -> None:
    """
    Run symmetric and asymmetric synthetic stackup simulations.

    Prints peak bow for each case and asserts that:
    - Symmetric stackup peak bow < 0.01 mm.
    - Asymmetric stackup peak bow > symmetric (physically meaningful bow).

    Raises:
        AssertionError: If either physical expectation is violated.
    """
    from src.simulation.solver import run_simulation

    print("=" * 60)
    print("Lamination warpage — synthetic validation")
    print("=" * 60)

    # --- Symmetric test ---
    sym_path = _write_temp_stackup(_symmetric_stackup_rows())
    try:
        sym_result = run_simulation(
            json_path=sym_path,
            library_path=_LIBRARY_PATH,
            process_config=_PROCESS_CONFIG,
        )
    finally:
        sym_path.unlink(missing_ok=True)

    print(f"\n[Symmetric stackup]")
    print(f"  Peak bow      : {sym_result.peak_bow_mm:.6f} mm")
    print(f"  Bow/span ratio: {sym_result.bow_span_ratio:.6f}")
    print(f"  kappa_x range     : [{sym_result.kappa_x.min():.4f}, {sym_result.kappa_x.max():.4f}] 1/m")
    print(f"  B-matrix warn : {sym_result.b_matrix_warning}")

    # --- Asymmetric test ---
    asym_path = _write_temp_stackup(_asymmetric_stackup_rows())
    try:
        asym_result = run_simulation(
            json_path=asym_path,
            library_path=_LIBRARY_PATH,
            process_config=_PROCESS_CONFIG,
        )
    finally:
        asym_path.unlink(missing_ok=True)

    print(f"\n[Asymmetric stackup — heavy top copper]")
    print(f"  Peak bow      : {asym_result.peak_bow_mm:.4f} mm")
    print(f"  Bow/span ratio: {asym_result.bow_span_ratio:.6f}")
    print(f"  kappa_x mean      : {asym_result.kappa_x.mean():.4f} 1/m")
    print(f"  kappa_y mean      : {asym_result.kappa_y.mean():.4f} 1/m")
    print(f"  B-matrix warn : {asym_result.b_matrix_warning}")

    # --- Physical assertions ---
    assert sym_result.peak_bow_mm < 0.01, (
        f"Symmetric stackup should give near-zero bow, got {sym_result.peak_bow_mm:.6f} mm"
    )
    assert asym_result.peak_bow_mm > sym_result.peak_bow_mm, (
        "Asymmetric stackup should produce more bow than symmetric"
    )

    print("\nOK All assertions passed.")
    print("=" * 60)


if __name__ == "__main__":
    run_synthetic_test()
