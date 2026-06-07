# Copper Balancing Application — Technical Summary

---

## 1. Project Objective

Predict and quantify PCB warpage using Classical Lamination Theory (CLT), and validate predictions against physical measurements taken by an Akrometrix shadow moiré system.

**Core question:** Can a simulation built from Gerber copper layout files and material properties predict the warpage shape and magnitude that is physically measured on fabricated boards?

---

## 2. System Architecture

```
Gerber Files (.gbr)
        │
        ▼
  Gerber Rasterizer (gerbv)
  → PNG copper density images
  → Compressed NPZ arrays (float32)
        │
        ▼
  Simulation Engine
  ├─ Imbalance Track   (copper asymmetry map)
  ├─ Density Track     (total copper map)
  └─ CLT Track         (full warpage prediction)
        │
        ▼
  Spatial Alignment    (rotation + resample)
        │
        ▼
  Comparison Metrics   (RMSE, R², Pearson, gradient angles)
        │
        ▼
  Sweep Results CSV    (all parameter combinations)
        │
        ▼
  Interactive Dashboard + 3D Fit Viewer
```

---

## 3. Physics: Classical Lamination Theory (CLT)

### 3.1 Laminate Model

The PCB is modelled as a symmetric/asymmetric laminated plate. Each layer is characterised by:
- **Young's modulus** E (GPa) — separately for Core and Prepreg forms
- **Poisson's ratio** ν
- **CTE** α_xy (in-plane), α_z (through-thickness) in ppm/°C
- **Glass transition temperature** Tg (°C)
- **Cure shrinkage** ε_cure (Prepreg only, ≈ 0.15% linear)

### 3.2 Full ABD Matrix Solver

The implementation uses the complete ABD stiffness matrix for an asymmetric laminate:

```
[N]   [A  B] [ε⁰]   [N_T]
[M] = [B  D] [κ ] - [M_T]
```

Where:
- **A_ij = Σ Q_ij · Δz**  (in-plane stiffness, N/m)
- **B_ij = Σ Q_ij · (z_top² − z_bot²) / 2**  (bending-extension coupling, N)
- **D_ij = Σ Q_ij · (z_top³ − z_bot³) / 3**  (bending stiffness, N·m)

For a free plate under thermal load (N = 0), the coupled equations yield:

**κ = (D − B·A⁻¹·B)⁻¹ · (M_T − B·A⁻¹·N_T)**

This Schur-complement form is the exact solution for asymmetric laminates.
Previous simplified form **κ = D⁻¹·M_T** (ignoring B) systematically underestimated curvature for real PCBs.

### 3.3 Thermal Loads

- **M_T** = thermal bending moment = Σ Q·α·ΔT·(z_top² − z_bot²)/2
- **N_T** = thermal in-plane force  = Σ Q·α·ΔT·Δz
- ΔT applied at layer level: Core/copper layers use (Tg_core − T_ambient), Prepreg uses (Tg_prepreg − T_ambient)

### 3.4 Curvature to Displacement

Curvature κ_x, κ_y is converted to out-of-plane displacement w via a Fast Fourier Transform (FFT) Poisson solver, which integrates the Kirchhoff plate equation spatially.

---

## 4. Material Database

8 named material sets covering the full test vehicle:

| Material | Tg (°C) | E_core (GPa) | E_prepreg (GPa) | CTE_xy_core | CTE_xy_prep |
|---|---|---|---|---|---|
| EM890K | 200 | 24.0 | 19.0 | 11.0 | 13.5 |
| IT988  | 200 | 25.0 | 20.0 | 12.0 | 14.5 |
| TU883  | 185 | 24.0 | 19.0 | 12.0 | 14.5 |
| M8000  | 185 | 23.0 | 18.0 | 12.0 | 14.5 |
| IT170  | 170 | 24.0 | 19.0 | 13.0 | 15.5 |
| EM528  | 170 | 22.0 | 17.0 | 14.0 | 16.5 |
| M1     | 155 | 22.0 | 17.0 | 13.0 | 15.5 |
| N4000-13 | 210 | 22.0 | 17.0 | 12.0 | 14.0 |

Core and Prepreg are differentiated:
- Core: fully cured, higher E, lower CTE (glass-fibre dominated)
- Prepreg: partially cured, lower E, higher CTE (resin dominated), non-zero cure shrinkage

---

## 5. Copper Density Maps

### Three Simulation Tracks

**Imbalance Track** (ΔT-independent):
- Computes weighted copper asymmetry about the neutral axis
- Each layer contributes: sign(z_mid) × oz_weight × copper_density(x,y)
- Positive = more copper above neutral axis → upward bow on cooling

**Density Track** (ΔT-independent):
- Unsigned sum of copper across all layers: Σ oz_weight × density(x,y)
- Identifies regions of high total copper mass

**CLT Track** (sweeps ΔT):
- Full warpage prediction in mm using the ABD solver
- Sweeps ΔT from TDR cure temperature down to ambient

### Raster Cache System

- Gerbers rendered at target DPI via gerbv → PNG
- PNGs converted to compressed float32 NPZ arrays
- Cache keyed as: `{board_name}_{gerber_quadrant}_{dpi}dpi/`
  - e.g., `Cu_Bal_TV_Q1_50dpi/`
- Lookup order: NPZ cache → PNG cache → render from Gerbers

---

## 6. Parameter Sweep

All parameter combinations are swept automatically across the full dataset.

### Swept Parameters

| Parameter | Description |
|---|---|
| DPI | Gerber rasterisation resolution (e.g. 50, 100, 200, 400) |
| ΔT | Temperature delta for CLT (°C) |
| Blur type | none / Gaussian / box |
| Gaussian σ | Blur kernel standard deviation (px) |
| Box radius | Box blur kernel half-width (px) |
| Fill missing | Biharmonic inpainting of measurement gaps |
| Denoise σ | Gaussian denoise applied to measurement |
| Crop fraction | Central region fraction used for comparison |

### Scale

- **16 manufacturers** × **2 sides** (Top/Bottom) × **4 quadrants** = 128 config entries
- Each config sweeps all parameter combinations
- Total combinations per config: DPI × ΔT × blur_ops × preprocessing × crop_fractions × dat_files
- All results written to a single shared CSV: `Cu_Bal_TV.csv`

### Resume System

- Each result row is keyed by all parameters + config name + dat file stem
- On restart, existing rows are loaded; only missing combinations are computed
- Handles schema migrations (e.g. added material column) automatically

---

## 7. Spatial Alignment

### Step 1: Rotation Correction
- Coarse angular search (±5°, 21 steps) over mask overlap score
- Refined with bounded scalar optimisation (scipy.optimize.minimize_scalar)
- Applied via ndimage rotate (reshape=False)

### Step 2: Grid Resampling
- Measurement resampled to simulation grid shape using bilinear interpolation (skimage)
- NaN propagation preserves measurement footprint

### Step 3: Kabsch/Umeyama 3D Rigid Transform
- Applied to the (x, y, z_displacement) point clouds after grid alignment
- Computes optimal rotation R, translation t, optional uniform scale s via SVD
- Provides rotation matrix R (9 components), translation vector t, and scale as diagnostic columns in CSV

---

## 8. Comparison Metrics

### Distance Metrics (3D and Z-only)

| Metric | Description |
|---|---|
| RMSE_Z | Root mean square of Z residual after linear regression (mm) |
| MAE_Z | Mean absolute Z error (mm) |
| P95_Z | 95th percentile absolute Z error (mm) |
| Max_Z | Maximum absolute Z error (mm) |
| RMSE_3D | Full 3D point-cloud distance RMSE |

Z metrics are computed on regression residuals (Z-scale matched), so they reflect shape mismatch independent of amplitude difference.

### Correlation Metrics

| Metric | Description | Target |
|---|---|---|
| Pearson r | Linear correlation of Z fields | → 1.0 |
| R² | Fraction of measurement variance explained | → 1.0 |
| Slope | Linear regression slope (sim vs meas) | → 1.0 |
| Intercept | Regression intercept | → 0.0 |

### Gradient Angle Metrics

Measures whether the **slope directions** (not just magnitudes) agree spatially.

- Both fields smoothed with Gaussian σ=3 px before gradient computation (removes pixel-level noise)
- Active pixels: gradient magnitude > 10% of 95th percentile (removes flat regions where direction is undefined)
- **angle_mean_deg**: mean angle between simulation and measurement gradient vectors
  - 0° = perfect directional agreement
  - 90° = uncorrelated (no signal)
  - 180° = consistent anti-correlation (sign flip or inverted convention)

### Magnitude Ratio

- mag_ratio = |∇_measurement| / |∇_simulation|
- Ratio < 1: simulation over-predicts local slopes
- Ratio > 1: simulation under-predicts local slopes

---

## 9. Dataset Coverage

### Board Design

- 22-layer PCB (layers L1–L22)
- Outer layers: 1 oz foil + plating
- Inner signal layers: 1 oz
- Inner power/ground planes: 2 oz (L9–L14)
- Dielectric: alternating Core (12 mil) and Prepreg (3 mil)

### Measurement Coverage

| Dimension | Count |
|---|---|
| Manufacturers | 16 (ACCL, AKM, DMC, FNDR, SCC, TTM, VGT, WUS) |
| Dielectric materials | 8 (EM890K, IT988, TU883, M8000, IT170, EM528, M1, N4000-13) |
| Board sides | 2 (Top, Bottom) |
| Quadrants | 4 (Q1, Q2, Q3, Q4) |
| Total configs | 128 |

---

## 10. Interactive Tools

### Sweep Results Dashboard (Dash/Plotly)

- Filter by material, manufacturer, source, side, quadrant
- Scatter / Line / Box plot of any metric vs any parameter
- Correlation heatmap across all numeric parameters and metrics
- "Best Configs" finder: rank parameter combinations by any metric
- Summary statistics panel

### 3D Fit Viewer (PyQt6 + Plotly)

- Per-row "View Fit" button in the results table
- Loads the exact Akrometrix .dat file used for that row
- Applies identical preprocessing (fill_missing, denoise) and alignment
- Renders three overlaid Plotly Surface traces:
  - **Blue**: Simulation displacement surface
  - **Red**: Measurement displacement surface
  - **RdBu**: Residual (Sim − Meas), toggled via legend
- NaN gaps filled via nearest-neighbour propagation for clean rendering
- HTML written to temp file (avoids QWebEngineView 2 MB setHtml limit)

---

## 11. Key Technical Challenges Solved

| Challenge | Solution |
|---|---|
| Asymmetric laminate underestimation | Full ABD matrix solver (Schur complement form) |
| Gradient angles always ~90° (noise) | Gaussian pre-smoothing (σ=3px) + 10% gradient magnitude threshold |
| CSV resume broken after material column added | Auto-migration + mixed-schema repair via csv module |
| fill_missing bool parsed wrong from CSV | `_parse_bool()` — `bool("False") == True` in Python |
| Large PNGs at high DPI | NPZ compressed float32 cache; PNG kept as secondary cache |
| Dashboard blank on large HTML | Write to temp file, load via QUrl (bypasses 2 MB setHtml limit) |
| Core vs Prepreg same properties | Differentiated material entries: `{MAT} Core` / `{MAT} Prepreg` |

---

## 12. Output CSV Schema

Each row represents one unique combination of parameters applied to one .dat measurement file.

**Identity columns:** `name`, `dat_file`, `location`, `side`, `material`, `dpi`, `source`, `blur_type`, `sigma`, `radius`, `delta_t_c`, `fill_missing`, `denoise_sigma`, `crop_fraction`, `akro_folder`

**Alignment diagnostics:** `scale`, `R00`–`R22`, `t_x`, `t_y`, `t_z`

**Distance metrics:** `rmse_3d`, `mae_3d`, `p95_3d`, `max_3d`, `rmse_z`, `mae_z`, `p95_z`, `max_z`

**Correlation:** `pearson_r`, `slope`, `intercept`, `r2`, `n`, `detrended`, `with_scaling`

**Gradient:** `angle_mean_deg`, `angle_median_deg`, `angle_p95_deg`, `mag_ratio_mean`, `mag_ratio_median`, `mag_ratio_p05`, `mag_ratio_p95`
