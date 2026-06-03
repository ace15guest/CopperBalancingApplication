# Lamination Warpage Simulation Pipeline

## Overview

The simulation predicts PCB warpage from thermal loading during lamination press using
Classical Lamination Theory (CLT). Each layer's copper density map produces spatially
varying stiffness and CTE fields; the D-matrix inversion yields a curvature field that
is integrated via FFT to produce the out-of-plane displacement surface w(x, y).

---

## Physics Summary

### Classical Lamination Theory

For a thin plate composed of N layers, the bending stiffness matrix is:

```
D_ij = Σ_k  Q_ij(x,y) · (z_top³ − z_bot³) / 3
```

The thermal bending moment resultant is:

```
MT_i = Σ_k  Q_ij · α_eff · ΔT_k · (z_top² − z_bot²) / 2
```

Curvature is obtained by inverting D at every grid cell:

```
κ = D⁻¹ · MT          (valid when coupling matrix B ≈ 0)
```

Displacement is reconstructed by solving the Poisson equation via FFT:

```
∇²w = κ_x + κ_y
```

### Rule of Mixtures

For copper layers with density map ρ(x, y) ∈ [0, 1]:

```
E_eff   = ρ · E_Cu  + (1−ρ) · E_FR4
α_eff   = ρ · α_Cu  + (1−ρ) · α_FR4
ν_eff   = ρ · ν_Cu  + (1−ρ) · ν_FR4
```

Dielectric layers use uniform material properties (ρ implicit = 1 for FR4).

### Thermal Loading

| Layer class   | ΔT used                          |
|---------------|----------------------------------|
| copper_outer  | T_press − T_ambient              |
| copper_inner  | T_press − T_ambient              |
| core          | T_press − T_ambient              |
| prepreg       | T_vitrification − T_ambient      |
| ignore (LPI)  | skipped                          |

Prepreg uses the reduced ΔT because resin only starts building stress after vitrification.

---

## Layer Classification

| Stackup material              | row_class      |
|-------------------------------|----------------|
| First / last copper row       | copper_outer   |
| All other copper rows         | copper_inner   |
| FR4 Core                      | core           |
| PrePreg 1080 / 1651 / 3080    | prepreg        |
| Liquid PhotoImageable Mask    | ignore         |

Copper outer layers always use 100 % density (no Gerber needed).
Copper inner layers load their density from the NPZ produced by the raster pipeline.

---

## Data Flow

```
JSON stackup
    │
    ▼
load_stackup()          → list[LayerSpec] (classified rows)
    │
    ▼
load_material_library() → dict[str, PhysicsProps]
    │
    ▼
resolve_layer_properties()  → density maps attached to LayerSpec
    │
    ▼
compute_z_coordinates()     → z_bot, z_top per layer (from neutral axis)
    │
    ▼
compute_effective_properties()  → E_eff, ν_eff, α_eff  (NY×NX arrays)
    │
    ├──▶ assemble_D_matrix()           → D (NY×NX×3×3)
    └──▶ compute_thermal_moments()     → MT (NY×NX×3)
                │
                ▼
          solve_curvature()            → κ_x, κ_y (NY×NX)
                │
                ▼
          reconstruct_displacement()   → w_mm (NY×NX)
                │
                ▼
          SimulationResult
```

---

## Implementation Files

| File                          | Contents                                     |
|-------------------------------|----------------------------------------------|
| `src/simulation/materials.py` | `PhysicsProps`, `load_material_library()`    |
| `src/simulation/solver.py`    | `LayerSpec`, `SimulationResult`, steps 1–10 |
| `src/simulation/validation.py`| Synthetic symmetric / asymmetric tests       |

---

## process_config Keys

```python
process_config = {
    "T_press_c":         float,   # press temperature [°C]
    "T_ambient_c":       float,   # room temperature [°C]
    "T_vitrification_c": float,   # prepreg vitrification temperature [°C]
    "board_width_m":     float,   # board physical width [m]
    "board_height_m":    float,   # board physical height [m]
    "default_grid":      (NY, NX) # fallback grid shape if no NPZ files found
}
```

---

## B-Matrix Diagnostic

The coupling matrix B = Σ Q · (z_top² − z_bot²) / 2 should be zero for a symmetric
laminate. A significant B indicates bending-extension coupling and means the simple
D⁻¹·MT solution underestimates actual curvature. The solver logs a warning when:

```
mean(||B||_F) / mean(||D||_F)  >  0.05
```
