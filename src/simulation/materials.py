"""
Material physics properties for the lamination warpage simulation.

Loads a named material set from the project's data/materials.json and
exposes properties in SI units for use by the CLT solver.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Maps stackup row material strings to the physics library key.
# All copper weight / foil-plating labels fall through to "Copper" by default.
STACKUP_TO_PHYSICS: dict[str, str] = {
    "FR4 Core":                   "FR4",
    "PrePreg 1080":               "FR4",
    "PrePreg 1651":               "FR4",
    "PrePreg 3080":               "FR4",
    "Liquid PhotoImageable Mask": "ignore",
}


@dataclass
class PhysicsProps:
    """
    Material physics properties in SI units for the CLT solver.

    Attributes:
        E_pa:          Isotropic Young's modulus [Pa] (used when Ex/Ey absent).
        nu:            Poisson's ratio [dimensionless].
        alpha_per_c:   Isotropic in-plane CTE [1/°C] (used when alpha_x/y absent).
        T_vit_c:       Vitrification / glass-transition temperature [°C], or None.
        Ex_pa:         Warp-direction Young's modulus [Pa] (orthotropic; None → isotropic).
        Ey_pa:         Weft-direction Young's modulus [Pa] (orthotropic; None → isotropic).
        alpha_x_per_c: Warp-direction CTE [1/°C] (orthotropic; None → isotropic).
        alpha_y_per_c: Weft-direction CTE [1/°C] (orthotropic; None → isotropic).
        cure_shrinkage: Linear cure shrinkage strain [dimensionless, positive = shrinkage].
                        Applied only to prepreg layers.
    """

    E_pa: float
    nu: float
    alpha_per_c: float
    T_vit_c: float | None = None
    Ex_pa: float | None = None
    Ey_pa: float | None = None
    alpha_x_per_c: float | None = None
    alpha_y_per_c: float | None = None
    cure_shrinkage: float = 0.0

    @property
    def eff_Ex(self) -> float:
        return self.Ex_pa if self.Ex_pa is not None else self.E_pa

    @property
    def eff_Ey(self) -> float:
        return self.Ey_pa if self.Ey_pa is not None else self.E_pa

    @property
    def eff_alpha_x(self) -> float:
        return self.alpha_x_per_c if self.alpha_x_per_c is not None else self.alpha_per_c

    @property
    def eff_alpha_y(self) -> float:
        return self.alpha_y_per_c if self.alpha_y_per_c is not None else self.alpha_per_c


def load_material_library(
    library_path: str | Path,
    set_name: str = "Set A",
) -> dict[str, PhysicsProps]:
    """
    Load material physics properties from the project's materials JSON file.

    Reads the named material set and returns a dict keyed by the physics
    material name (e.g. "FR4", "Copper") with all values converted to SI
    units: Young's modulus → Pa (×10⁹), CTE → 1/°C (×10⁻⁶).

    Args:
        library_path: Path to the materials.json file.
        set_name:     Name of the material set to load (e.g. "Set A").

    Returns:
        Dict mapping physics material name → PhysicsProps.

    Raises:
        FileNotFoundError: If library_path does not exist.
        ValueError:        If set_name is not found in the library.
    """
    library_path = Path(library_path)
    raw = json.loads(library_path.read_text(encoding="utf-8"))

    target = next(
        (s for s in raw["material_sets"] if s["name"] == set_name), None
    )
    if target is None:
        available = [s["name"] for s in raw["material_sets"]]
        raise ValueError(
            f"Material set '{set_name}' not found. Available: {available}"
        )

    result: dict[str, PhysicsProps] = {}
    for m in target["materials"]:
        result[m["name"]] = PhysicsProps(
            E_pa=m["youngs_modulus_gpa"] * 1e9,
            nu=m["poissons_ratio"],
            alpha_per_c=m["cte_xy_ppm_c"] * 1e-6,
            T_vit_c=m.get("tg_c"),
            Ex_pa=m["youngs_modulus_x_gpa"] * 1e9 if "youngs_modulus_x_gpa" in m else None,
            Ey_pa=m["youngs_modulus_y_gpa"] * 1e9 if "youngs_modulus_y_gpa" in m else None,
            alpha_x_per_c=m["cte_x_ppm_c"] * 1e-6 if "cte_x_ppm_c" in m else None,
            alpha_y_per_c=m["cte_y_ppm_c"] * 1e-6 if "cte_y_ppm_c" in m else None,
            cure_shrinkage=m.get("cure_shrinkage_linear_pct", 0.0) / 100.0,
        )
    return result


def physics_name_for(material: str) -> str:
    """
    Map a stackup row material string to its physics library key.

    Dielectric materials resolve via STACKUP_TO_PHYSICS.  Any copper weight
    or foil-plating label (e.g. "1 oz", "foil + plating 2.0") resolves to
    "Copper".  Unrecognised strings also resolve to "Copper".

    Args:
        material: Stackup row material string.

    Returns:
        Physics material name: "FR4", "Copper", or "ignore".
    """
    if material in STACKUP_TO_PHYSICS:
        return STACKUP_TO_PHYSICS[material]
    if material.endswith(" Core") or material.endswith(" Prepreg"):
        # Return the full name so Core and Prepreg can have different properties.
        # e.g. "890K Core" → "890K Core",  "IT988 Prepreg" → "IT988 Prepreg"
        return material
    return "Copper"
