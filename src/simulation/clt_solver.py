"""CLT pipeline entry point — runs from in-memory Stackup and density maps."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.models import CopperDensityMap, SimResult, Stackup
from src.simulation.materials import load_material_library, physics_name_for
from src.simulation.solver import (
    LayerSpec,
    _classify_dielectric,
    assemble_D_matrix,
    compute_thermal_moments,
    compute_z_coordinates,
    reconstruct_displacement,
    solve_curvature,
)

_MATERIALS_JSON = Path(__file__).parents[2] / "data" / "materials.json"
_MIL_TO_M = 25.4e-6

logger = logging.getLogger(__name__)


def solve_clt(
    stackup: Stackup,
    density_maps: list[CopperDensityMap],
    delta_temp_c: float,
    pixel_size_m: float = 25.4e-3 / 300,
) -> SimResult:
    """Run the CLT pipeline on an in-memory Stackup and pre-rasterised density maps.

    Args:
        stackup:      Board stackup definition.
        density_maps: Copper density maps, one per inner copper layer with a Gerber.
        delta_temp_c: ΔT applied uniformly to all layers [°C].
        pixel_size_m: Physical size of one pixel [m].  Defaults to 300 DPI pitch.

    Returns:
        SimResult with displacement [mm], x/y coords in mm, and mode="clt".
    """
    # --- Classify rows into LayerSpec objects ---
    rows = stackup.rows
    copper_indices = [i for i, r in enumerate(rows) if r.row_type == "copper"]
    outer_set = {copper_indices[0], copper_indices[-1]} if copper_indices else set()

    layers: list[LayerSpec] = []
    for i, row in enumerate(rows):
        if row.row_type == "copper":
            row_class = "copper_outer" if i in outer_set else "copper_inner"
        else:
            row_class = _classify_dielectric(row.material)
        layers.append(LayerSpec(
            row_class=row_class,
            thickness_m=row.finish_thickness_mil * _MIL_TO_M,
            material=row.material,
            gerber_path=row.gerber_path,
            layer_number=row.layer_number,
        ))

    # --- Material library ---
    library = load_material_library(_MATERIALS_JSON, set_name=stackup.material_set_name)
    cu = library.get("Copper")

    # --- Grid shape from first density map ---
    grid_shape: tuple[int, int] = (200, 200)
    if density_maps:
        grid_shape = density_maps[0].density.shape

    # Index density maps by layer label ("L2", "L3", …)
    dm_by_label: dict[str, CopperDensityMap] = {dm.layer_name: dm for dm in density_maps}

    # --- Attach material props and density maps ---
    for layer in layers:
        if layer.row_class == "ignore":
            continue

        phys_name = physics_name_for(layer.material)
        props = library.get(phys_name)
        if props is None:
            logger.warning("No library entry for '%s' (→ '%s') — skipping.", layer.material, phys_name)
            continue

        layer.E_pa          = props.E_pa
        layer.nu            = props.nu
        layer.alpha_per_c   = props.alpha_per_c
        layer.delta_T       = delta_temp_c
        layer.Ex_pa         = props.Ex_pa
        layer.Ey_pa         = props.Ey_pa
        layer.alpha_x_per_c = props.alpha_x_per_c
        layer.alpha_y_per_c = props.alpha_y_per_c
        layer.cure_shrinkage = props.cure_shrinkage

        if layer.row_class == "copper_outer":
            layer.density_map = np.ones(grid_shape, dtype=np.float32)
            if cu:
                layer.E_pa, layer.nu, layer.alpha_per_c = cu.E_pa, cu.nu, cu.alpha_per_c

        elif layer.row_class == "copper_inner":
            label = f"L{layer.layer_number}"
            dm = dm_by_label.get(label)
            if dm is not None:
                density = dm.density.astype(np.float32)
                if density.shape != grid_shape:
                    from PIL import Image
                    img = Image.fromarray(density)
                    img = img.resize((grid_shape[1], grid_shape[0]), Image.BILINEAR)
                    density = np.array(img, dtype=np.float32)
                layer.density_map = density
            else:
                logger.warning("No density map for L%s — using 50%% fill.", layer.layer_number)
                layer.density_map = np.full(grid_shape, 0.5, dtype=np.float32)
            if cu:
                layer.E_pa, layer.nu, layer.alpha_per_c = cu.E_pa, cu.nu, cu.alpha_per_c

    # --- Board physical dimensions ---
    NY, NX = grid_shape
    board_width_m  = NX * pixel_size_m
    board_height_m = NY * pixel_size_m
    logger.debug(
        "CLT grid %d×%d px, pixel=%.1f µm → board %.1f × %.1f mm",
        NX, NY, pixel_size_m * 1e6, board_width_m * 1e3, board_height_m * 1e3,
    )

    # --- CLT steps 4–9 ---
    delta_T_map = {"core": delta_temp_c, "prepreg": delta_temp_c}
    z_coords  = compute_z_coordinates(layers)
    D         = assemble_D_matrix(layers, z_coords, grid_shape, library)
    MT        = compute_thermal_moments(layers, z_coords, delta_T_map, grid_shape, library)
    kappa_x, kappa_y, _ = solve_curvature(D, MT, layers, z_coords, grid_shape, library)
    w_mm      = reconstruct_displacement(kappa_x, kappa_y, board_width_m, board_height_m)

    x_coords = np.arange(NX, dtype=np.float32) * float(pixel_size_m * 1e3)
    y_coords = np.arange(NY, dtype=np.float32) * float(pixel_size_m * 1e3)

    return SimResult(mode="clt", displacement=w_mm, x_coords=x_coords, y_coords=y_coords)
