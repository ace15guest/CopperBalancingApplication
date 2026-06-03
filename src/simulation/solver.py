"""
CLT-based lamination warpage solver.

Implements a ten-step pipeline:
  1.  load_stackup              – parse JSON, classify rows
  2.  load_material_library     – load E / α / ν in SI units
  3.  resolve_layer_properties  – attach material props + density maps
  4.  compute_z_coordinates     – z_bot / z_top from neutral axis
  5.  compute_effective_properties – rule-of-mixtures per grid cell
  6.  assemble_D_matrix         – bending stiffness (NY×NX×3×3)
  7.  compute_thermal_moments   – MT (NY×NX×3)
  8.  solve_curvature           – κ_x, κ_y via D⁻¹·MT
  9.  reconstruct_displacement  – FFT Poisson solver → w [mm]
  10. run_simulation            – orchestrator, returns SimulationResult
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.simulation.materials import (
    PhysicsProps,
    load_material_library,
    physics_name_for,
)

logger = logging.getLogger(__name__)

_MIL_TO_M = 25.4e-6   # 1 mil → meters
_UM_TO_M  = 1e-6       # 1 µm  → meters
_B_WARNING_THRESHOLD = 0.05  # ||B||/||D|| ratio above which a warning is issued

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

_PREPREG_MATERIALS = {"PrePreg 1080", "PrePreg 1651", "PrePreg 3080"}
_CORE_MATERIAL     = "FR4 Core"
_IGNORE_MATERIAL   = "Liquid PhotoImageable Mask"


def _classify_dielectric(material: str) -> str:
    """Return 'core', 'prepreg', or 'ignore' for a dielectric material string."""
    if material == _IGNORE_MATERIAL:
        return "ignore"
    if material == _CORE_MATERIAL or material.endswith(" Core"):
        return "core"
    if material in _PREPREG_MATERIALS or material.endswith(" Prepreg"):
        return "prepreg"
    return "ignore"


@dataclass
class LayerSpec:
    """
    Fully described layer for the CLT solver.

    Attributes:
        row_class:    Classification string: copper_outer | copper_inner |
                      core | prepreg | ignore.
        thickness_m:  Layer thickness in metres.
        material:     Original material string from the stackup JSON.
        gerber_path:  Path to the source Gerber file (copper rows only).
        layer_number: Copper layer index (1-based), or None for dielectrics.
        density_map:  Float32 (NY, NX) array with values 0–1, or None for
                      dielectrics.  Populated by resolve_layer_properties().
        E_pa:         Base Young's modulus [Pa] from the material library.
        nu:           Base Poisson's ratio from the material library.
        alpha_per_c:  Base in-plane CTE [1/°C] from the material library.
        delta_T:      Applied temperature change [°C].  Set by run_simulation.
    """

    row_class: str
    thickness_m: float
    material: str
    gerber_path: Path | None = None
    layer_number: int | None = None
    density_map: NDArray | None = None
    E_pa: float | None = None
    nu: float | None = None
    alpha_per_c: float | None = None
    delta_T: float | None = None
    Ex_pa: float | None = None
    Ey_pa: float | None = None
    alpha_x_per_c: float | None = None
    alpha_y_per_c: float | None = None
    cure_shrinkage: float = 0.0


@dataclass
class SimulationResult:
    """
    Output of run_simulation().

    Attributes:
        w_mm:             Out-of-plane displacement [mm], shape (NY, NX).
        kappa_x:          Curvature in x [1/m], shape (NY, NX).
        kappa_y:          Curvature in y [1/m], shape (NY, NX).
        peak_bow_mm:      max(w) − min(w) [mm].
        bow_span_ratio:   peak_bow_mm / board_diagonal_mm (IPC-style).
        b_matrix_warning: True if the B-matrix asymmetry exceeded the
                          threshold, indicating coupling terms were ignored.
        layer_metadata:   List of dicts summarising each active layer.
    """

    w_mm: NDArray
    kappa_x: NDArray
    kappa_y: NDArray
    peak_bow_mm: float
    bow_span_ratio: float
    b_matrix_warning: bool
    layer_metadata: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step 1 — load_stackup
# ---------------------------------------------------------------------------

def load_stackup(json_path: str | Path) -> list[LayerSpec]:
    """
    Parse a stackup JSON file and classify every row as a LayerSpec.

    Supports both v1 (finish_thickness_um) and v2 (finish_thickness_mil)
    JSON schemas.  Copper rows are classified as copper_outer (first and last
    copper layer) or copper_inner.  Dielectric rows become core, prepreg, or
    ignore.

    Args:
        json_path: Path to a stackup JSON file saved by the application.

    Returns:
        List of LayerSpec objects in stackup order (top to bottom).

    Raises:
        FileNotFoundError: If json_path does not exist.
        ValueError:        If the JSON contains no recognised rows.
    """
    json_path = Path(json_path)
    raw = json.loads(json_path.read_text(encoding="utf-8"))

    rows = raw.get("stackup", [])
    if not rows:
        raise ValueError(f"No stackup rows found in {json_path}")

    # Resolve thickness → metres (handle both schema versions)
    def _thickness_m(row: dict) -> float:
        if "finish_thickness_mil" in row:
            return row["finish_thickness_mil"] * _MIL_TO_M
        return row.get("finish_thickness_um", 0.0) * _UM_TO_M

    # First pass: collect all copper rows to identify outer layers
    copper_indices = [
        i for i, r in enumerate(rows) if r.get("row_type") == "copper"
    ]
    outer_set = set()
    if copper_indices:
        outer_set = {copper_indices[0], copper_indices[-1]}

    specs: list[LayerSpec] = []
    for i, row in enumerate(rows):
        material = row.get("material", "")
        gerber = (
            Path(row["gerber_path"]) if row.get("gerber_path") else None
        )
        layer_num = row.get("layer_number")
        thickness = _thickness_m(row)

        if row.get("row_type") == "copper":
            row_class = "copper_outer" if i in outer_set else "copper_inner"
        else:
            row_class = _classify_dielectric(material)
            if row_class == "ignore" and material not in (_IGNORE_MATERIAL, ""):
                logger.warning("Unrecognised material '%s' — treating as ignore.", material)

        specs.append(LayerSpec(
            row_class=row_class,
            thickness_m=thickness,
            material=material,
            gerber_path=gerber,
            layer_number=layer_num,
        ))

    return specs


# ---------------------------------------------------------------------------
# Step 2 — load_material_library  (re-exported from materials module)
# ---------------------------------------------------------------------------
# Consumers import load_material_library from this module for convenience.
# The actual implementation lives in src/simulation/materials.py.


# ---------------------------------------------------------------------------
# Step 3 — resolve_layer_properties
# ---------------------------------------------------------------------------

def resolve_layer_properties(
    layers: list[LayerSpec],
    library: dict[str, PhysicsProps],
    npz_folder: str | Path | None = None,
    default_grid: tuple[int, int] = (200, 200),
) -> list[LayerSpec]:
    """
    Attach material properties and density maps to every active LayerSpec.

    Rules:
    - copper_outer: density_map = ones(grid_shape) — 100 % copper coverage.
    - copper_inner: density_map loaded from the NPZ file whose stem matches
                    the Gerber stem in npz_folder.  Falls back to 0.5
                    (uniform 50 %) with a warning if the file is missing.
    - core / prepreg: density_map = None (uniform dielectric).
    - ignore: unchanged.

    The canonical grid shape is determined from the first successfully loaded
    NPZ file.  If no NPZ is found, default_grid is used.

    Args:
        layers:       Classified LayerSpec list from load_stackup().
        library:      Dict of PhysicsProps from load_material_library().
        npz_folder:   Folder containing *.npz density files.  May be None if
                      the stackup has no inner copper layers with Gerbers.
        default_grid: (NY, NX) fallback grid shape when no NPZ files exist.

    Returns:
        The same list with .density_map, .E_pa, .nu, .alpha_per_c populated.
    """
    npz_folder = Path(npz_folder) if npz_folder else None

    # Determine canonical grid shape from the first loadable NPZ.
    grid_shape: tuple[int, int] = default_grid
    for layer in layers:
        if layer.row_class == "copper_inner" and layer.gerber_path and npz_folder:
            candidate = npz_folder / f"{layer.gerber_path.stem}.npz"
            if candidate.exists():
                data = np.load(candidate)
                grid_shape = data["density"].shape
                break

    fr4  = library.get("FR4")
    cu   = library.get("Copper")

    for layer in layers:
        if layer.row_class == "ignore":
            continue

        phys_name = physics_name_for(layer.material)
        props: PhysicsProps | None = library.get(phys_name)
        if props is None:
            logger.warning(
                "No library entry for '%s' (resolved to '%s') — skipping.",
                layer.material, phys_name,
            )
            continue

        layer.E_pa          = props.E_pa
        layer.nu            = props.nu
        layer.alpha_per_c   = props.alpha_per_c
        layer.Ex_pa         = props.Ex_pa
        layer.Ey_pa         = props.Ey_pa
        layer.alpha_x_per_c = props.alpha_x_per_c
        layer.alpha_y_per_c = props.alpha_y_per_c
        layer.cure_shrinkage = props.cure_shrinkage

        if layer.row_class == "copper_outer":
            layer.density_map = np.ones(grid_shape, dtype=np.float32)

        elif layer.row_class == "copper_inner":
            loaded = False
            if npz_folder and layer.gerber_path:
                npz_path = npz_folder / f"{layer.gerber_path.stem}.npz"
                if npz_path.exists():
                    data = np.load(npz_path)
                    dm = data["density"].astype(np.float32)
                    if dm.shape != grid_shape:
                        from PIL import Image  # lazy import — only needed for resize
                        img = Image.fromarray(dm)
                        img = img.resize(
                            (grid_shape[1], grid_shape[0]), Image.BILINEAR
                        )
                        dm = np.array(img, dtype=np.float32)
                    layer.density_map = dm
                    loaded = True
            if not loaded:
                warnings.warn(
                    f"No NPZ found for copper_inner layer {layer.layer_number} "
                    f"(gerber_path={layer.gerber_path}) — using 50 % density.",
                    stacklevel=2,
                )
                layer.density_map = np.full(grid_shape, 0.5, dtype=np.float32)
            # Override base props to Copper (density map handles mixing in step 5)
            layer.E_pa        = cu.E_pa
            layer.nu          = cu.nu
            layer.alpha_per_c = cu.alpha_per_c

    return layers


# ---------------------------------------------------------------------------
# Step 4 — compute_z_coordinates
# ---------------------------------------------------------------------------

def compute_z_coordinates(
    layers: list[LayerSpec],
) -> list[tuple[float, float]]:
    """
    Compute z_bot and z_top for every active layer measured from the neutral axis.

    Ignored layers (LPI mask) are excluded from the z-stack entirely.
    The neutral axis is placed at the geometric mid-plane of the remaining
    stack.

    Args:
        layers: LayerSpec list (may contain ignore-class entries).

    Returns:
        List of (z_bot, z_top) tuples in metres, one per active layer, in
        the same order as the active subset of *layers*.  Ignored layers are
        omitted.
    """
    active = [l for l in layers if l.row_class != "ignore"]
    total_thickness = sum(l.thickness_m for l in active)
    mid = total_thickness / 2.0

    z = -mid
    coords: list[tuple[float, float]] = []
    for layer in active:
        z_bot = z
        z_top = z + layer.thickness_m
        coords.append((z_bot, z_top))
        z = z_top

    return coords


# ---------------------------------------------------------------------------
# Step 5 — compute_effective_properties
# ---------------------------------------------------------------------------

def compute_effective_properties(
    layer: LayerSpec,
    grid_shape: tuple[int, int],
    library: dict[str, PhysicsProps],
) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
    """
    Compute spatially varying effective material properties for one layer.

    For copper layers the rule of mixtures is applied cell-by-cell using
    the layer's density map.  For dielectric layers uniform scalar arrays
    (broadcasting the library values) are returned.

    Orthotropic dielectric properties (Ex ≠ Ey, alpha_x ≠ alpha_y) are used
    when present in the library; otherwise the isotropic values are used for
    both x and y.  Copper is always treated as isotropic.

    Args:
        layer:      A LayerSpec with density_map / E_pa / nu / alpha_per_c set.
        grid_shape: (NY, NX) output array shape.
        library:    Dict of PhysicsProps from load_material_library().

    Returns:
        Tuple (Ex_eff, Ey_eff, nu_eff, alpha_x_eff, alpha_y_eff) each as
        float64 (NY, NX) array.
    """
    dielectric = library.get("FR4") or next(
        (v for k, v in library.items() if k != "Copper"), None
    )
    cu = library.get("Copper")
    NY, NX = grid_shape

    if layer.density_map is not None and dielectric is not None and cu is not None:
        rho = layer.density_map.astype(np.float64)
        Ex_eff      = rho * cu.E_pa        + (1.0 - rho) * dielectric.eff_Ex
        Ey_eff      = rho * cu.E_pa        + (1.0 - rho) * dielectric.eff_Ey
        nu_eff      = rho * cu.nu          + (1.0 - rho) * dielectric.nu
        alpha_x_eff = rho * cu.alpha_per_c + (1.0 - rho) * dielectric.eff_alpha_x
        alpha_y_eff = rho * cu.alpha_per_c + (1.0 - rho) * dielectric.eff_alpha_y
    else:
        ex = layer.Ex_pa         if layer.Ex_pa         is not None else layer.E_pa
        ey = layer.Ey_pa         if layer.Ey_pa         is not None else layer.E_pa
        ax = layer.alpha_x_per_c if layer.alpha_x_per_c is not None else layer.alpha_per_c
        ay = layer.alpha_y_per_c if layer.alpha_y_per_c is not None else layer.alpha_per_c
        Ex_eff      = np.full((NY, NX), ex,         dtype=np.float64)
        Ey_eff      = np.full((NY, NX), ey,         dtype=np.float64)
        nu_eff      = np.full((NY, NX), layer.nu,   dtype=np.float64)
        alpha_x_eff = np.full((NY, NX), ax,         dtype=np.float64)
        alpha_y_eff = np.full((NY, NX), ay,         dtype=np.float64)

    return Ex_eff, Ey_eff, nu_eff, alpha_x_eff, alpha_y_eff


# ---------------------------------------------------------------------------
# Helpers: Q matrix entries from E and ν arrays
# ---------------------------------------------------------------------------

def _q_matrix(
    Ex: NDArray, Ey: NDArray, nu: NDArray
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Compute plane-stress reduced stiffness for an orthotropic layer.

    For isotropic material pass Ex == Ey; the result then reduces to the
    standard isotropic Q-matrix with Q11 == Q22.

    Assumes νyx = ν * Ey/Ex (Lekhnitskii reciprocal relation with νxy = ν).

    Args:
        Ex: Warp-direction Young's modulus [Pa], shape (NY, NX).
        Ey: Weft-direction Young's modulus [Pa], shape (NY, NX).
        nu: In-plane Poisson's ratio νxy, shape (NY, NX).

    Returns:
        Tuple (Q11, Q12, Q22, Q66) each of shape (NY, NX).
    """
    denom = 1.0 - nu ** 2 * (Ey / Ex)
    Q11 = Ex / denom
    Q22 = Ey / denom
    Q12 = nu * Ey / denom
    Q66 = Ex / (2.0 * (1.0 + nu))
    return Q11, Q12, Q22, Q66


# ---------------------------------------------------------------------------
# Step 6 — assemble_D_matrix
# ---------------------------------------------------------------------------

def assemble_D_matrix(
    layers: list[LayerSpec],
    z_coords: list[tuple[float, float]],
    grid_shape: tuple[int, int],
    library: dict[str, PhysicsProps],
) -> NDArray:
    """
    Assemble the bending stiffness matrix D for the full laminate.

    D_ij(x, y) = Σ_k  Q_ij_k(x,y) · (z_top_k³ − z_bot_k³) / 3

    All copper layers contribute spatially varying Q via the rule of mixtures;
    dielectric layers contribute uniform Q.  No Python loops over grid cells —
    operations are fully vectorised over (NY, NX).

    Args:
        layers:      Active LayerSpec list (ignored layers already removed in
                     z_coords).
        z_coords:    (z_bot, z_top) list from compute_z_coordinates().
        grid_shape:  (NY, NX).
        library:     Dict of PhysicsProps.

    Returns:
        D as float64 (NY, NX, 3, 3) array [Pa·m³ = N·m].
        Indices: (y, x, 0, 0)=D11, (0,1)=D12, (1,1)=D22, (2,2)=D66.
    """
    NY, NX = grid_shape
    D = np.zeros((NY, NX, 3, 3), dtype=np.float64)

    active = [l for l in layers if l.row_class != "ignore"]
    for layer, (z_bot, z_top) in zip(active, z_coords):
        if layer.E_pa is None:
            continue  # library entry missing — skip silently

        Ex_eff, Ey_eff, nu_eff, _, _ = compute_effective_properties(layer, grid_shape, library)
        Q11, Q12, Q22, Q66 = _q_matrix(Ex_eff, Ey_eff, nu_eff)

        dz3 = (z_top ** 3 - z_bot ** 3) / 3.0
        D[..., 0, 0] += Q11 * dz3
        D[..., 0, 1] += Q12 * dz3
        D[..., 1, 0] += Q12 * dz3
        D[..., 1, 1] += Q22 * dz3
        D[..., 2, 2] += Q66 * dz3

    return D


# ---------------------------------------------------------------------------
# Step 7 — compute_thermal_moments
# ---------------------------------------------------------------------------

def compute_thermal_moments(
    layers: list[LayerSpec],
    z_coords: list[tuple[float, float]],
    delta_T_map: dict[str, float],
    grid_shape: tuple[int, int],
    library: dict[str, PhysicsProps],
) -> NDArray:
    """
    Compute the thermal bending moment resultant MT.

    MT_i(x, y) = Σ_k  [Q_ij · α_eff · ΔT_k] · (z_top² − z_bot²) / 2

    The ΔT applied to each layer depends on its row_class:
        copper_outer / copper_inner / core → delta_T_map["core"]
        prepreg                            → delta_T_map["prepreg"]

    Args:
        layers:       Active LayerSpec list.
        z_coords:     (z_bot, z_top) from compute_z_coordinates().
        delta_T_map:  Dict with keys "core" and "prepreg" mapping to ΔT [°C].
        grid_shape:   (NY, NX).
        library:      Dict of PhysicsProps.

    Returns:
        MT as float64 (NY, NX, 3) array [Pa·m² = N/m = N·m/m].
        Index 0 → MT_x, 1 → MT_y, 2 → MT_xy (always zero for isotropic).
    """
    NY, NX = grid_shape
    MT = np.zeros((NY, NX, 3), dtype=np.float64)

    active = [l for l in layers if l.row_class != "ignore"]
    for layer, (z_bot, z_top) in zip(active, z_coords):
        if layer.E_pa is None:
            continue

        if layer.row_class == "prepreg":
            dT = delta_T_map.get("prepreg", 0.0)
        else:
            dT = delta_T_map.get("core", 0.0)

        # Cure shrinkage: negative eigenstrain (shrinkage) locked in at cure; prepreg only.
        cure_eps = -layer.cure_shrinkage if layer.row_class == "prepreg" else 0.0

        if dT == 0.0 and cure_eps == 0.0:
            continue

        Ex_eff, Ey_eff, nu_eff, alpha_x_eff, alpha_y_eff = compute_effective_properties(
            layer, grid_shape, library
        )
        Q11, Q12, Q22, _ = _q_matrix(Ex_eff, Ey_eff, nu_eff)

        dz2 = (z_top ** 2 - z_bot ** 2) / 2.0

        if dT != 0.0:
            MT[..., 0] += (Q11 * alpha_x_eff + Q12 * alpha_y_eff) * dT * dz2
            MT[..., 1] += (Q12 * alpha_x_eff + Q22 * alpha_y_eff) * dT * dz2

        if cure_eps != 0.0:
            # Isotropic chemical shrinkage: ε_x = ε_y = cure_eps
            MT[..., 0] += (Q11 + Q12) * cure_eps * dz2
            MT[..., 1] += (Q12 + Q22) * cure_eps * dz2

    return MT


# ---------------------------------------------------------------------------
# Step 8 — solve_curvature
# ---------------------------------------------------------------------------

def solve_curvature(
    D: NDArray,
    MT: NDArray,
    layers: list[LayerSpec],
    z_coords: list[tuple[float, float]],
    grid_shape: tuple[int, int],
    library: dict[str, PhysicsProps],
) -> tuple[NDArray, NDArray, bool]:
    """
    Compute the curvature field by inverting D at every grid cell.

    κ(x, y) = D⁻¹(x, y) · MT(x, y)

    Also computes the B-matrix (bending-extension coupling) as a diagnostic.
    A non-zero B means the simple D⁻¹·MT solution is approximate; a warning
    is logged and flagged in the return value when ||B||_F / ||D||_F > 0.05.

    Args:
        D:          Bending stiffness (NY, NX, 3, 3) from assemble_D_matrix().
        MT:         Thermal moments (NY, NX, 3) from compute_thermal_moments().
        layers:     Active LayerSpec list (for B-matrix computation).
        z_coords:   (z_bot, z_top) from compute_z_coordinates().
        grid_shape: (NY, NX).
        library:    Dict of PhysicsProps.

    Returns:
        Tuple (kappa_x, kappa_y, b_matrix_warning):
            kappa_x / kappa_y — curvature arrays (NY, NX) [1/m].
            b_matrix_warning  — True if B asymmetry was significant.
    """
    # Invert D at every grid cell (np.linalg.inv operates on last two dims)
    D_inv = np.linalg.inv(D)
    kappa = np.einsum("...ij,...j->...i", D_inv, MT)  # (NY, NX, 3)
    kappa_x = kappa[..., 0]
    kappa_y = kappa[..., 1]

    # B-matrix diagnostic — B = Σ Q · (z_top² − z_bot²) / 2
    NY, NX = grid_shape
    B = np.zeros((NY, NX, 3, 3), dtype=np.float64)
    active = [l for l in layers if l.row_class != "ignore"]
    for layer, (z_bot, z_top) in zip(active, z_coords):
        if layer.E_pa is None:
            continue
        Ex_eff, Ey_eff, nu_eff, _, _ = compute_effective_properties(layer, grid_shape, library)
        Q11, Q12, Q22, Q66 = _q_matrix(Ex_eff, Ey_eff, nu_eff)
        dz2 = (z_top ** 2 - z_bot ** 2) / 2.0
        B[..., 0, 0] += Q11 * dz2
        B[..., 0, 1] += Q12 * dz2
        B[..., 1, 0] += Q12 * dz2
        B[..., 1, 1] += Q22 * dz2
        B[..., 2, 2] += Q66 * dz2

    b_norm = np.mean(np.linalg.norm(B.reshape(NY, NX, 9), axis=-1))
    d_norm = np.mean(np.linalg.norm(D.reshape(NY, NX, 9), axis=-1))
    ratio = b_norm / (d_norm + 1e-30)
    b_warning = ratio > _B_WARNING_THRESHOLD
    if b_warning:
        logger.warning(
            "B-matrix asymmetry is significant (||B||/||D|| = %.3f > %.2f). "
            "Curvature prediction may be underestimated.",
            ratio, _B_WARNING_THRESHOLD,
        )

    return kappa_x, kappa_y, b_warning


# ---------------------------------------------------------------------------
# Step 9 — reconstruct_displacement
# ---------------------------------------------------------------------------

def reconstruct_displacement(
    kappa_x: NDArray,
    kappa_y: NDArray,
    board_width_m: float,
    board_height_m: float,
) -> NDArray:
    """
    Reconstruct the out-of-plane displacement surface from the curvature field.

    Solves the Poisson equation ∇²w = κ_x + κ_y via FFT:

        Ŵ = − FFT(κ_x + κ_y) / (k_x² + k_y²)

    The DC component is forced to zero (rigid-body translation removed) and
    the mean of the result is also subtracted to set the reference plane.

    Args:
        kappa_x:        Curvature in x [1/m], shape (NY, NX).
        kappa_y:        Curvature in y [1/m], shape (NY, NX).
        board_width_m:  Physical board width [m] (x-direction).
        board_height_m: Physical board height [m] (y-direction).

    Returns:
        w_mm: Out-of-plane displacement [mm], shape (NY, NX).
    """
    NY, NX = kappa_x.shape

    # Spatial frequency grids [rad/m]
    dx = board_width_m  / NX
    dy = board_height_m / NY
    kx = 2.0 * np.pi * np.fft.fftfreq(NX, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(NY, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    k2 = KX ** 2 + KY ** 2

    source = kappa_x + kappa_y
    F = np.fft.fft2(source)

    k2[0, 0] = 1.0        # prevent division by zero; DC will be zeroed anyway
    W = -F / k2
    W[0, 0] = 0.0         # remove rigid-body translation (mean = 0)

    w = np.real(np.fft.ifft2(W))
    w -= w.mean()          # ensure zero mean reference plane

    return w * 1e3  # metres → mm


# ---------------------------------------------------------------------------
# Step 10 — run_simulation
# ---------------------------------------------------------------------------

def run_simulation(
    json_path: str | Path,
    library_path: str | Path,
    process_config: dict,
    npz_folder: str | Path | None = None,
) -> SimulationResult:
    """
    End-to-end lamination warpage simulation orchestrator.

    Calls steps 1–9 in order and returns a fully populated SimulationResult.

    process_config must contain:
        T_press_c         – lamination press temperature [°C]
        T_ambient_c       – ambient / final temperature [°C]
        T_vitrification_c – prepreg vitrification temperature [°C]
        board_width_m     – physical board width [m]
        board_height_m    – physical board height [m]

    Optional process_config keys:
        material_set      – name of the material set to load (default "Set A")
        default_grid      – (NY, NX) fallback grid if no NPZ files exist
                            (default (200, 200))

    Args:
        json_path:      Path to the stackup JSON file.
        library_path:   Path to the materials.json library file.
        process_config: Dict of process parameters (see above).
        npz_folder:     Optional folder containing *.npz density files for
                        inner copper layers.

    Returns:
        SimulationResult with displacement map, curvature, bow metrics, and
        per-layer metadata.
    """
    # --- Unpack process parameters ---
    T_press   = float(process_config["T_press_c"])
    T_ambient = float(process_config["T_ambient_c"])
    T_vit     = float(process_config["T_vitrification_c"])
    bw        = float(process_config["board_width_m"])
    bh        = float(process_config["board_height_m"])
    set_name  = process_config.get("material_set", "Set A")
    default_grid = tuple(process_config.get("default_grid", (200, 200)))

    delta_T_map = {
        "core":    T_press - T_ambient,
        "prepreg": T_vit   - T_ambient,
    }

    # Step 1 — parse stackup
    layers = load_stackup(json_path)

    # Step 2 — load material library
    library = load_material_library(library_path, set_name=set_name)

    # Step 3 — resolve properties + density maps
    layers = resolve_layer_properties(
        layers, library, npz_folder=npz_folder, default_grid=default_grid
    )

    # Determine grid shape from resolved density maps
    grid_shape: tuple[int, int] = default_grid
    for layer in layers:
        if layer.density_map is not None:
            grid_shape = layer.density_map.shape
            break

    # Assign ΔT to each layer
    for layer in layers:
        if layer.row_class in ("copper_outer", "copper_inner", "core"):
            layer.delta_T = T_press - T_ambient
        elif layer.row_class == "prepreg":
            layer.delta_T = T_vit - T_ambient

    # Step 4 — z-coordinates
    z_coords = compute_z_coordinates(layers)

    # Step 5–6 — assemble D
    D = assemble_D_matrix(layers, z_coords, grid_shape, library)

    # Step 7 — thermal moments
    MT = compute_thermal_moments(
        layers, z_coords, delta_T_map, grid_shape, library
    )

    # Step 8 — solve curvature
    kappa_x, kappa_y, b_warning = solve_curvature(
        D, MT, layers, z_coords, grid_shape, library
    )

    # Step 9 — reconstruct displacement
    w_mm = reconstruct_displacement(kappa_x, kappa_y, bw, bh)

    # Bow metrics
    peak_bow_mm = float(w_mm.max() - w_mm.min())
    diagonal_mm = np.sqrt((bw * 1e3) ** 2 + (bh * 1e3) ** 2)
    bow_span_ratio = peak_bow_mm / diagonal_mm if diagonal_mm > 0 else 0.0

    # Layer metadata
    active = [l for l in layers if l.row_class != "ignore"]
    metadata = [
        {
            "row_class":    l.row_class,
            "material":     l.material,
            "thickness_um": l.thickness_m * 1e6,
            "layer_number": l.layer_number,
            "delta_T":      l.delta_T,
            "has_density":  l.density_map is not None,
        }
        for l in active
    ]

    return SimulationResult(
        w_mm=w_mm,
        kappa_x=kappa_x,
        kappa_y=kappa_y,
        peak_bow_mm=peak_bow_mm,
        bow_span_ratio=bow_span_ratio,
        b_matrix_warning=b_warning,
        layer_metadata=metadata,
    )
