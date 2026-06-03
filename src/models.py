from dataclasses import dataclass, field
from pathlib import Path
from numpy.typing import NDArray


@dataclass
class MaterialProperties:
    name: str
    youngs_modulus_gpa: float
    poissons_ratio: float
    cte_xy_ppm_c: float
    cte_z_ppm_c: float
    tg_c: float | None = None


@dataclass
class MaterialSet:
    name: str
    materials: list[MaterialProperties] = field(default_factory=list)

    def get(self, material_name: str) -> MaterialProperties | None:
        for m in self.materials:
            if m.name == material_name:
                return m
        return None


@dataclass
class StackupRow:
    row_type: str                       # "copper" | "dielectric"
    layer_number: int | None = None     # copper rows only, auto-assigned
    gerber_path: Path | None = None     # copper rows only
    material: str = ""                  # dielectric: "{set} Core/Prepreg"; copper: oz/foil label
    construction: str = ""              # dielectric: glass style, e.g. "1080 — 76%"
    finish_thickness_mil: float = 0.0


@dataclass
class Stackup:
    rows: list[StackupRow] = field(default_factory=list)
    lamination_profile: list[tuple[float, float]] = field(default_factory=list)  # (temp_C, time_min)
    material_set_name: str = "Set A"


@dataclass
class CopperDensityMap:
    layer_name: str
    density: NDArray        # 2D array, values 0.0–1.0
    x_coords: NDArray       # 1D
    y_coords: NDArray       # 1D


@dataclass
class MeasurementData:
    source_file: str
    displacement: NDArray   # 2D interpolated grid (mm)
    x_coords: NDArray
    y_coords: NDArray


@dataclass
class SimResult:
    mode: str               # "clt" or "hifi"
    displacement: NDArray   # 2D array (mm)
    x_coords: NDArray
    y_coords: NDArray


@dataclass
class GradientMetrics:
    angle_mean_deg: float
    angle_median_deg: float
    angle_p95_deg: float
    mag_ratio_mean: float
    mag_ratio_median: float
    mag_ratio_p05: float
    mag_ratio_p95: float


@dataclass
class ComparisonMetrics:
    rms_error: float
    r_squared: float
    pearson: float
    gradient_correlation: float
    hotspot_overlap: float
    ipc_bow_ratio: float
