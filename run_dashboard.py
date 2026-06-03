"""Launch the Sweep Analysis Dashboard.

Usage:
    python run_dashboard.py
    python run_dashboard.py path/to/results.csv

Then open: http://127.0.0.1:8050
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent
_DEFAULT_CSV = _ROOT / "assets" / "sweep_results"

from dashboard import run_dashboard


def _find_csv() -> Path | None:
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists():
            return p
        print(f"File not found: {p}")
        return None
    # Find most recently modified CSV in sweep_results/
    csvs = sorted(_DEFAULT_CSV.glob("*.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
    if csvs:
        return csvs[0]
    return None


if __name__ == "__main__":
    csv = _find_csv()
    if csv is None:
        print("No sweep results CSV found.")
        print(f"Run a sweep first, or pass a CSV path as an argument.")
        sys.exit(1)
    try:
        run_dashboard(data_path=str(csv), debug=False)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
