# ΔT Sweep Pipeline

## Inputs

| Input | What it is |
|-------|-----------|
| **Board name** | Used to key the PNG/NPZ cache folders on disk |
| **Akrometrix folder** | Directory of `.dat` files — one per measurement location/side (e.g. Q1 Top, Q2 Bottom) |
| **Stackup JSON** | Layer order, thicknesses, material set, and Gerber file paths for each copper layer |

---

## Step 1 — Rasterise Gerbers → Density Maps (per DPI)

Each copper layer Gerber is rendered to a PNG at the chosen DPI, then converted to a floating-point density map (0.0–1.0 per pixel) stored as an NPZ file.

- Cached on disk — if the NPZ folder already exists for a given DPI it is reused, so re-running the sweep at the same DPI costs nothing.
- Outer copper layers (L1, L22) are assumed 100 % filled — no Gerber needed.
- Inner layers with missing Gerbers default to 50 % fill with a warning.

**Output:** one density map per copper layer, shape `(NY, NX)`.

---

## Step 2 — Preprocess Akrometrix Measurements (once, up front)

All `.dat` files are loaded raw (μm → mm, sentinel value 9999 masked to NaN), then each (fill_missing, denoise_sigma) combination is applied and cached in memory.

- **Fill missing (biharmonic):** solves ∇⁴u = 0 to predict NaN pixels from surrounding curvature. Slow — done once per file per combination, not repeated per DPI/ΔT/blur.
- **Denoise σ:** large-median pre-filter → MAD outlier clip → Gaussian smooth. Removes soldermask/copper step-function offsets.
- σ = 0 means no denoise.

**Output:** preprocessed MeasurementData grid per (file, fill_missing, denoise_sigma).

---

## Step 3 — CLT Solve (per DPI × ΔT)

Classical Lamination Theory is run using the density maps from Step 1 and the material properties from the stackup's material set.

```
w(x,y) [mm]  =  f( density_maps, E, α, ν, ΔT )
```

- **ΔT** is applied uniformly to all layers. The sweep finds which ΔT produces a warpage map that best matches the Akrometrix measurement.
- The CLT solve is skipped entirely for a (DPI, ΔT) pair if all its downstream combinations are already in the results CSV (resume support).

**Output:** displacement map `w(x,y)` in mm, same shape as the density maps.

---

## Step 4 — Apply Blur to CLT Output (per blur operation)

The CLT displacement map is optionally smoothed before comparison:

| blur_type | What it does |
|-----------|-------------|
| `clt` | No post-processing — raw CLT output |
| `gaussian` | Gaussian filter with standard deviation σ px |
| `box` | Uniform (box) filter with half-width radius px |

**Scheduling:** `clt` runs at **every ΔT**. `gaussian` and `box` run only at the **midpoint ΔT** — because blur parameters are ΔT-independent, the best σ/radius found at the midpoint applies across all ΔT values.

**Output:** blurred displacement map, same shape as CLT output.

---

## Step 5 — Spatial Alignment: Akro → CLT Grid (per blur op × preprocessed Akro)

`alignment.align()` registers each preprocessed Akro measurement onto the CLT grid:

1. **Rotation search (±5°):** finds the small CCW rotation that maximises mask overlap — corrects for how the board was loaded in the Akrometrix machine.
2. **Resample:** bilinear resample of the Akro from its native pixel dimensions to match the CLT grid shape exactly.

After this step both arrays share the same `(NY, NX)` pixel grid and x,y coordinates.

---

## Step 6 — Comparison Metrics (per crop fraction)

An optional centre-crop is applied to both arrays (e.g. 80 % keeps only the inner 80 % of each dimension, discarding edge effects).

Then `align_and_compare()` computes:

**Z-only metrics** (most useful — directly measure warpage prediction error):

| Column | Meaning | Good value |
|--------|---------|-----------|
| `rmse_z` | Root mean square of CLT − Akro displacement error (mm) | lower |
| `mae_z` | Mean absolute error (mm) — less sensitive to outliers than RMSE | lower |
| `p95_z` | 95th-percentile absolute error — captures the worst 5 % | lower |
| `pearson_r` | Correlation of CLT vs Akro displacement fields (−1 to 1) | closer to 1 |
| `r2` | Fraction of Akro variance explained by CLT (0 to 1) | closer to 1 |
| `slope` | Linear fit slope — 1.0 means CLT amplitude matches Akro | closer to 1 |
| `intercept` | Z-offset between CLT and Akro after detrending | closer to 0 |

**Gradient metrics** — measure whether the shape (slope directions) agree, not just the amplitude:

| Column | Meaning | Good value |
|--------|---------|-----------|
| `angle_mean_deg` | Mean angular difference between CLT and Akro gradient vectors | lower |
| `mag_ratio_mean` | Mean ratio of CLT gradient magnitude to Akro gradient magnitude | closer to 1 |

> **Note on the Kabsch rotation matrix (R00–R22):** The Kabsch 3D rigid-body fit always returns R ≈ identity because both arrays are already on the same pixel grid (identical x,y coordinates). The R columns are not informative and can be ignored. The useful Kabsch outputs are `scale` and `t_z`.

---

## Output — One CSV Row Per Combination

Each row in the results CSV uniquely identifies one experiment:

```
(DPI, ΔT, blur_type, sigma, radius, fill_missing, denoise_sigma, crop_fraction, dat_file)
```

The sweep is **resumable** — completed rows are written incrementally to the CSV and skipped on re-run.

---

## Sweep Combination Count

```
Total rows = N_dpi
           × [ N_dt × N_clt_ops  +  N_blur_ops ]   ← blur optimisation
           × N_fill_missing
           × N_denoise_sigma
           × N_crop_fractions
           × N_dat_files
```

### Example

- 2 DPI × 10 ΔT × (1 clt + 2 gaussian + 2 box) × 2 fill × 2 denoise × 2 crop × 5 .dat files
- CLT rows: 2 × 10 × 1 × 2 × 2 × 2 × 5 = 800
- Blur rows: 2 × 1 × 4 × 2 × 2 × 2 × 5 = 320
- **Total: 1 120 rows**

---

## Decision Workflow

```
1. Find best ΔT     → filter blur_type = "clt", minimise rmse_z across ΔT
2. Find best blur    → filter dt = midpoint, compare clt / gaussian / box rmse_z
3. Find best preproc → compare fill_missing and denoise_sigma combinations
4. Find best DPI     → compare rmse_z across DPI values at fixed best ΔT
5. Validate          → check pearson_r and angle_mean_deg at the winning config
```
