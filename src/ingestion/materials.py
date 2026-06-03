import json
from pathlib import Path
from src.models import MaterialProperties, MaterialSet

_MATERIALS_FILE = Path(__file__).parents[2] / "data" / "materials.json"


def load_material_sets() -> list[MaterialSet]:
    """Load all named material sets from data/materials.json."""
    raw = json.loads(_MATERIALS_FILE.read_text(encoding="utf-8"))
    sets = []
    for s in raw["material_sets"]:
        materials = [
            MaterialProperties(
                name=m["name"],
                youngs_modulus_gpa=m["youngs_modulus_gpa"],
                poissons_ratio=m["poissons_ratio"],
                cte_xy_ppm_c=m["cte_xy_ppm_c"],
                cte_z_ppm_c=m["cte_z_ppm_c"],
                tg_c=m.get("tg_c"),
            )
            for m in s["materials"]
        ]
        sets.append(MaterialSet(name=s["name"], materials=materials))
    return sets


def get_material_set(name: str) -> MaterialSet | None:
    """Return the named material set, or None if not found."""
    for ms in load_material_sets():
        if ms.name == name:
            return ms
    return None
