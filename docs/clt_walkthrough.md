# CLT Solver — Step by Step

Classical Lamination Theory (CLT) is borrowed from the aerospace composite materials world
and adapted here to predict PCB warpage from thermal loading.  This document walks through
every step of `src/simulation/solver.py` in plain language, explaining *what* is being
computed and *why*.

---

## The Core Idea

A PCB is a stack of copper and FR4 layers bonded together.  When the board cools from the
lamination press temperature back to room temperature, every layer tries to shrink.  The
problem is that copper and FR4 shrink by different amounts per degree:

| Material | CTE (in-plane) |
|----------|---------------|
| Copper   | ~17 ppm/°C    |
| FR4      | ~14 ppm/°C    |

Because the layers are glued together, they can't shrink independently — they fight each
other.  If the copper is perfectly balanced above and below the board's midplane, the
forces cancel and the board stays flat.  If the copper is heavier on one side, that side
pulls harder and the board bows.

CLT turns this intuition into numbers by treating the board as a stack of elastic sheets
and computing the resulting curvature from first principles.

---

## Step 1 — Load the Stackup

**Code:** `load_stackup(json_path)`

The stackup JSON is read and every row is classified into one of five categories:

| Class | Meaning |
|-------|---------|
| `copper_outer` | Top-most or bottom-most copper layer |
| `copper_inner` | All other copper layers |
| `core` | FR4 core dielectric |
| `prepreg` | Prepreg dielectric (PP 1080, 1651, 3080) |
| `ignore` | LPI solder mask — mechanically negligible, excluded |

The outer copper layers are identified as the first and last copper rows in the JSON,
regardless of layer number.  Everything in between is inner copper.

**Why does the outer/inner distinction matter?**  Outer layers are assumed to have 100%
copper coverage (no Gerber file needed) because they are typically solid foil before
etching.  Inner layers get their real copper density from a rasterised Gerber image.

---

## Step 2 — Load the Material Library

**Code:** `load_material_library(library_path, set_name)`

Properties for FR4 and Copper are loaded from `data/materials.json` and converted to SI
units:

- Young's modulus **E** [Pa] — how stiff the material is (resistance to stretching)
- Poisson's ratio **ν** [–] — how much it squeezes sideways when you pull it lengthwise
- Thermal expansion **α** [1/°C] — how much it grows per degree of temperature rise

The material set (Set A, Set B, …) lets you swap in different supplier specifications
without changing the code.

---

## Step 3 — Resolve Layer Properties + Density Maps

**Code:** `resolve_layer_properties(layers, library, npz_folder)`

Each active layer gets its material properties attached.  For copper layers the density
map (a 2D array of values 0–1 indicating how much of each grid cell is copper) is loaded
from the pre-processed `.npz` files.

**Rule of mixtures** is used later (Step 5) to blend copper and FR4 properties at each
grid cell based on the density value:

```
E_eff  = ρ · E_Cu  + (1 − ρ) · E_FR4
α_eff  = ρ · α_Cu  + (1 − ρ) · α_FR4
```

Where ρ is the local copper density (0 = pure FR4, 1 = solid copper).

If an inner copper layer has no NPZ file, the solver falls back to ρ = 0.5 (50% uniform
density) and logs a warning.

### How copper weight (oz) is handled

The density map **only captures areal coverage** — it answers "what fraction of this grid
cell has copper?".  It does not know whether that copper is 1 oz or 2 oz.

Copper weight is captured separately through `finish_thickness_mil` in the stackup JSON,
which sets the layer's physical thickness (`thickness_m`).  The two inputs work together:

| Input | Source | What it captures |
|-------|--------|-----------------|
| Density map ρ (0–1) | NPZ from Gerber | Lateral copper pattern — *where* the copper is |
| `finish_thickness_mil` → `thickness_m` | Stackup JSON | Copper weight — *how thick* that copper is |

A 2 oz layer has `finish_thickness_mil = 2.8`, making it twice as thick as a 1 oz layer
at `finish_thickness_mil = 1.4`.  This difference flows into the z-coordinates computed
in Step 4 and then into the D-matrix and thermal moment integrals in Steps 6–7 — with
non-linear consequences explained there.

---

## Step 4 — Compute Z-Coordinates

**Code:** `compute_z_coordinates(layers)`

The solver needs to know how far each layer is from the board's **neutral axis** — the
plane about which the board bends.

The neutral axis is placed at the **geometric midplane** of the active stack (total
thickness / 2).  Ignored layers (LPI mask) do not contribute to the thickness.

Each active layer gets a `z_bot` and `z_top` coordinate in metres measured from the
neutral axis:

```
z = 0  →  neutral axis (midplane)
z > 0  →  above midplane (top half)
z < 0  →  below midplane (bottom half)
```

**Why does z-position matter?**  A layer that is far from the neutral axis has more
mechanical leverage — the same force creates more bending moment the further it is from
the centre.  This is the same reason an I-beam is stiffer than a solid rod of the same
cross-section.

---

## Step 5 — Compute Effective Properties (Rule of Mixtures)

**Code:** `compute_effective_properties(layer, grid_shape, library)`

For each layer, a 2D array of effective E, ν, and α values is computed.  For copper
layers this is done cell-by-cell using the density map:

```
E_eff(x,y)  = ρ(x,y) · E_Cu  + (1 − ρ(x,y)) · E_FR4
α_eff(x,y)  = ρ(x,y) · α_Cu  + (1 − ρ(x,y)) · α_FR4
```

For dielectric layers (core, prepreg) the properties are uniform — a constant array
filled with the library value.

The result is that every grid cell has its own effective stiffness and expansion
coefficient, spatially reflecting the actual Gerber copper pattern.

---

## Step 6 — Assemble the D-Matrix (Bending Stiffness)

**Code:** `assemble_D_matrix(layers, z_coords, grid_shape, library)`

The **D-matrix** is the board's resistance to bending.  Think of it as the PCB equivalent
of the flexural rigidity EI from beam theory, but extended to a 2D plate.

For each layer k, the plane-stress reduced stiffness coefficients Q are computed from
E and ν:

```
Q11 = E / (1 − ν²)      (stiffness along x or y)
Q12 = ν·E / (1 − ν²)    (coupling between x and y)
Q66 = E / (2·(1 + ν))   (shear stiffness)
```

These are summed across all layers, weighted by their position in the z-stack:

```
D_ij = Σ_k  Q_ij_k(x,y) · (z_top_k³ − z_bot_k³) / 3
```

The **cubic** dependence on z means copper weight has a non-linear impact on stiffness.
For a layer of thickness h centred at distance z_mid from the neutral axis:

```
(z_top³ − z_bot³) / 3  ≈  z_mid² · h
```

So the D contribution scales with **both z-position squared and layer thickness**.

| Scenario | D contribution |
|----------|---------------|
| 1 oz layer at z_mid = 1 mm | z_mid² × h = 1² × 1 = 1× |
| 2 oz layer at same z_mid   | z_mid² × 2h = 1² × 2 = **2×** |
| 1 oz layer at z_mid = 2 mm | z_mid² × h = 4² × 1 = **4×** |
| 2 oz layer at z_mid = 2 mm | z_mid² × 2h = 4 × 2 = **8×** |

This is why outer copper layers dominate board stiffness even when they are only 1 oz —
they are far from the neutral axis.  Moving a 2 oz layer from the centre to the outside
of the stack can increase its D contribution by an order of magnitude.

The result is a (NY × NX × 3 × 3) array — a 3×3 stiffness matrix at every grid cell.

---

## Step 7 — Compute Thermal Moments

**Code:** `compute_thermal_moments(layers, z_coords, delta_T_map, grid_shape, library)`

The **thermal moment** MT is the bending load that the temperature change applies to the
board.  It plays the same role as an applied moment in beam bending theory, but it comes
from thermal expansion mismatch rather than an external load.

Two different ΔT values are applied depending on the layer type:

| Layer type | ΔT used | Physical reason |
|-----------|---------|----------------|
| Copper, Core | T_press − T_ambient | These were already solid at press temperature; they store the full thermal strain on cooling |
| Prepreg | T_vitrification − T_ambient | Prepreg is liquid/rubbery above its glass transition temperature (Tg) and can't store stress.  It only locks in strain once it vitrifies at Tg |

The thermal moment is computed as:

```
MT_i(x,y) = Σ_k  [Q_ij_k · α_eff_k · ΔT_k] · (z_top_k² − z_bot_k²) / 2
```

For a layer of thickness h at z_mid this simplifies to approximately `Q · α · ΔT · z_mid · h`.
The contribution scales with **both z-position and layer thickness**:

| Scenario | MT contribution (relative) |
|----------|--------------------------|
| 1 oz layer at z_mid = 1 mm | z_mid × h = 1 × 1 = 1× |
| 2 oz layer at same z_mid   | z_mid × 2h = 1 × 2 = **2×** |
| 1 oz layer at z_mid = 2 mm | z_mid × h = 2 × 1 = **2×** |
| 2 oz layer at z_mid = 2 mm | z_mid × 2h = 2 × 2 = **4×** |

This is the key to understanding copper balancing.  A 2 oz plane layer buried near the
board's midplane creates far less thermal moment than the same layer near the surface —
even though it has twice the copper.  **Where** the copper sits matters as much as
**how much** copper there is.

---

## Step 8 — Solve for Curvature

**Code:** `solve_curvature(D, MT, ...)`

Now we have both the bending stiffness (D) and the thermal loading (MT).  The curvature
is simply:

```
κ(x,y) = D⁻¹(x,y) · MT(x,y)
```

This is inverted at every grid cell independently — a vectorised matrix inversion over
the full (NY × NX) grid.

**κ_x** is the curvature in the x-direction (like a cylinder curving along x).
**κ_y** is the curvature in the y-direction.
Positive κ means the board is concave upward (bowl shape).

### The B-Matrix Warning

The solver also computes the **B-matrix** (bending-extension coupling) as a diagnostic:

```
B_ij = Σ_k  Q_ij_k · (z_top_k² − z_bot_k²) / 2
```

For a perfectly symmetric stackup B = 0.  When B is non-zero the board's bending and
in-plane stretching are coupled — pulling on it also bends it.  The simple D⁻¹·MT
solution ignores this coupling, which means the curvature prediction is a **lower bound**
on the true warpage.

The solver flags a warning when `||B|| / ||D|| > 0.05` (5%).  For the test board this
ratio is around 5.7 even after fixing the layer thicknesses, because the copper *patterns*
on mirror layers differ (a plane layer vs a signal layer at the same depth creates
different local B contributions).

---

## Step 9 — Reconstruct the Displacement Surface

**Code:** `reconstruct_displacement(kappa_x, kappa_y, board_width_m, board_height_m)`

Curvature κ and displacement w are related by:

```
∇²w = κ_x + κ_y
```

This is a **Poisson equation** — the same equation that governs heat conduction, fluid
pressure, and electrostatics.  Solving it gives the out-of-plane shape of the board.

The solver uses the **FFT method** for efficiency:

1. Compute the 2D FFT of the curvature source field (κ_x + κ_y)
2. Divide by the spatial frequency squared in Fourier space: `Ŵ = −F̂ / (k_x² + k_y²)`
3. The DC component (k=0) is forced to zero to remove rigid-body translation
4. Inverse FFT back to get w(x, y)
5. Subtract the mean of w to set the reference plane to zero

The result is w_mm — the out-of-plane displacement in millimetres at every grid cell.
Positive values mean the board surface is above the reference plane; negative values are
below it.

---

## Step 10 — Bow Metrics

**Code:** computed inside `run_simulation()`

Two summary numbers are derived from the displacement map:

**Peak bow** — the total range of the displacement surface:
```
peak_bow_mm = max(w) − min(w)
```
This is the worst-case height difference between the highest and lowest point on the board.

**Bow/span ratio** — normalised by the board diagonal (IPC-7711 style):
```
bow_span_ratio = peak_bow_mm / board_diagonal_mm
```
IPC-7711 specifies a maximum bow/span ratio of 0.0075 (0.75%) for most assembly
processes.  The test board result of ~0.00034 (0.034%) is well within this limit.

---

## Assumptions and Limitations

| Assumption | Effect if violated |
|-----------|-------------------|
| Isotropic materials (E same in x and y) | FR4 is slightly anisotropic; prepreg warp/fill directions differ.  Error is typically < 10%. |
| Linear elasticity (small strains) | Valid for typical PCB bow; breaks down for severely warped boards. |
| Geometric neutral axis = mechanical neutral axis | Conservative approximation; true neutral axis shifts with asymmetric copper loading. |
| B-matrix coupling ignored | Result is a lower bound.  Significant for asymmetric copper patterns. |
| Uniform ΔT through the thickness | Real lamination has a temperature gradient during press; modelling this requires a time-resolved solver. |
| Gerber DPI adequate to capture copper features | 50 DPI resolves features to ~0.5 mm.  Fine traces and small vias are averaged out. |

---

## Quick Reference: What Each Variable Means

| Symbol | Name | Unit | Shape in code |
|--------|------|------|--------------|
| ρ | Copper density | — | (NY, NX) |
| E | Young's modulus | Pa | scalar per layer |
| ν | Poisson's ratio | — | scalar per layer |
| α | CTE (in-plane) | 1/°C | scalar per layer |
| Q11, Q12, Q66 | Plane-stress stiffness | Pa | (NY, NX) |
| z_bot, z_top | Layer z-coordinates from neutral axis | m | scalar per layer |
| D | Bending stiffness matrix | N·m | (NY, NX, 3, 3) |
| MT | Thermal moment resultant | N | (NY, NX, 3) |
| κ_x, κ_y | Curvature | 1/m | (NY, NX) |
| w | Out-of-plane displacement | m → mm | (NY, NX) |
