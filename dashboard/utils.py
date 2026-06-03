"""Utility functions for the sweep dashboard."""

from typing import Any, Dict, List
import numpy as np
import pandas as pd


def format_column_name(col: str) -> str:
    formatted = col.replace("_", " ").title()
    replacements = {
        "Rmse": "RMSE", "Mae": "MAE", "P95": "P95", "Dpi": "DPI",
        "R2": "R²", "Deg": "(deg)", "3D": "3D", "Mag": "Magnitude",
        "Clt": "CLT", "T C": "ΔT °C",
    }
    for old, new in replacements.items():
        formatted = formatted.replace(old, new)
    return formatted


def get_metric_description(metric: str) -> str:
    descriptions = {
        "rmse_z":          "RMSE of z residuals after z-scale matching (shape error, mm)",
        "mae_z":           "Mean absolute z error after z-scale matching (mm)",
        "p95_z":           "95th-percentile absolute z error after z-scale matching (mm)",
        "max_z":           "Maximum absolute z error after z-scale matching (mm)",
        "rmse_3d":         "RMS 3D point-cloud distance after Kabsch alignment",
        "mae_3d":          "Mean absolute 3D distance after Kabsch alignment",
        "pearson_r":       "Pearson correlation of source z vs Akro z (shape agreement, −1 to 1)",
        "r2":              "R² — fraction of Akro z-variance explained by source",
        "slope":           "Linear regression slope: Akro ≈ slope × source + intercept. 1.0 = perfect amplitude match",
        "intercept":       "Linear regression intercept (z-offset after slope correction)",
        "angle_mean_deg":  "Mean angular difference between source and Akro gradient vectors (lower = better)",
        "mag_ratio_mean":  "Mean gradient magnitude ratio source/Akro (1.0 = perfect amplitude match)",
    }
    return descriptions.get(metric, metric)


def create_filter_options(df: pd.DataFrame, column: str) -> List[Dict[str, Any]]:
    if column not in df.columns:
        return []
    unique_vals = df[column].dropna().unique()
    try:
        unique_vals = sorted(unique_vals)
    except TypeError:
        unique_vals = sorted(str(v) for v in unique_vals)
    return [{"label": str(v), "value": v} for v in unique_vals]


def get_color_scale(n: int) -> List[str]:
    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    ]
    return (colors * ((n // len(colors)) + 1))[:n]
