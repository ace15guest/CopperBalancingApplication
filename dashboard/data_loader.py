"""Data loader for the sweep results CSV."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class DataLoader:
    # Sweep identity / parameter columns
    PARAMETER_COLUMNS = [
        "name", "location", "side", "material",
        "dpi", "source", "blur_type", "sigma", "radius",
        "delta_t_c", "fill_missing", "denoise_sigma", "crop_fraction",
    ]

    # Output metric columns
    METRIC_COLUMNS = [
        "rmse_z", "mae_z", "p95_z", "max_z",
        "rmse_3d", "mae_3d", "p95_3d", "max_3d",
        "pearson_r", "slope", "intercept", "r2",
        "angle_mean_deg", "angle_median_deg", "angle_p95_deg",
        "mag_ratio_mean", "mag_ratio_median", "mag_ratio_p05", "mag_ratio_p95",
    ]

    NUMERIC_PARAMS = ["dpi", "sigma", "radius", "delta_t_c", "denoise_sigma", "crop_fraction"]
    CATEGORICAL_PARAMS = ["name", "location", "side", "material", "source", "blur_type", "fill_missing"]

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.df: pd.DataFrame = pd.DataFrame()
        self._load()

    def _load(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")
        try:
            self.df = pd.read_csv(self.file_path)
        except Exception:
            # Fall back to skipping malformed rows (e.g. mixed schema during migration)
            self.df = pd.read_csv(self.file_path, on_bad_lines="skip", engine="python")
            print(f"Warning: some rows skipped due to schema mismatch in {self.file_path.name}. "
                  "Run a sweep to repair the file.")
        for col in self.METRIC_COLUMNS + self.NUMERIC_PARAMS:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        if "fill_missing" in self.df.columns:
            self.df["fill_missing"] = self.df["fill_missing"].astype(str)
        print(f"Loaded {len(self.df)} rows from {self.file_path.name}")

    def get_parameter_columns(self) -> List[str]:
        return [c for c in self.PARAMETER_COLUMNS if c in self.df.columns]

    def get_metric_columns(self) -> List[str]:
        return [c for c in self.METRIC_COLUMNS if c in self.df.columns]

    def get_numeric_parameters(self) -> List[str]:
        return [c for c in self.NUMERIC_PARAMS if c in self.df.columns]

    def get_categorical_parameters(self) -> List[str]:
        return [c for c in self.CATEGORICAL_PARAMS if c in self.df.columns]

    def filter_data(self, filters: Dict[str, Any]) -> pd.DataFrame:
        df = self.df.copy()
        for col, val in filters.items():
            if col not in df.columns or val is None or val == []:
                continue
            if isinstance(val, list):
                df = df[df[col].isin(val)]
            else:
                df = df[df[col] == val]
        return df

    def get_summary_statistics(self, column: str,
                                filters: Optional[Dict] = None) -> Dict[str, float]:
        df = self.filter_data(filters) if filters else self.df
        if column not in df.columns:
            return {}
        data = df[column].dropna()
        if len(data) == 0:
            return {}
        return {
            "count":  int(len(data)),
            "mean":   float(data.mean()),
            "std":    float(data.std()),
            "min":    float(data.min()),
            "q25":    float(data.quantile(0.25)),
            "median": float(data.median()),
            "q75":    float(data.quantile(0.75)),
            "max":    float(data.max()),
        }

    def find_best_configs(self, metric: str, top_n: int = 10,
                          minimize: bool = True,
                          filters: Optional[Dict] = None) -> pd.DataFrame:
        df = self.filter_data(filters) if filters else self.df
        if metric not in df.columns:
            return pd.DataFrame()
        df_sorted = df.dropna(subset=[metric]).sort_values(metric, ascending=minimize)
        cols = self.get_parameter_columns() + [metric]
        cols = [c for c in cols if c in df_sorted.columns]
        return df_sorted.head(top_n)[cols]

    def get_correlation_matrix(self, columns: Optional[List[str]] = None,
                                filters: Optional[Dict] = None) -> pd.DataFrame:
        df = self.filter_data(filters) if filters else self.df
        if columns is None:
            columns = self.get_numeric_parameters() + self.get_metric_columns()
        columns = [c for c in columns if c in df.columns]
        return df[columns].corr()
