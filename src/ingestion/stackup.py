import json
from pathlib import Path
from src.models import Stackup, StackupRow

_VERSION = 2


def save_stackup(stackup: Stackup, file_path: Path) -> None:
    """Persist a stackup to a JSON file."""
    data = {
        "version": _VERSION,
        "material_set": stackup.material_set_name,
        "stackup": [
            {
                "row_type":             row.row_type,
                "layer_number":         row.layer_number,
                "material":             row.material,
                "finish_thickness_mil": row.finish_thickness_mil,
                "gerber_path":          str(row.gerber_path) if row.gerber_path else None,
            }
            for row in stackup.rows
        ],
    }
    Path(file_path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_stackup(file_path: Path) -> Stackup:
    """Load a stackup from a JSON file."""
    raw = json.loads(Path(file_path).read_text(encoding="utf-8"))
    rows = [
        StackupRow(
            row_type=r["row_type"],
            layer_number=r.get("layer_number"),
            material=r.get("material", ""),
            # support v1 files that stored µm — convert to mils on load
            finish_thickness_mil=r.get("finish_thickness_mil", r.get("finish_thickness_um", 0.0) / 25.4),
            gerber_path=Path(r["gerber_path"]) if r.get("gerber_path") else None,
        )
        for r in raw["stackup"]
    ]
    return Stackup(rows=rows, material_set_name=raw.get("material_set", "Set A"))
