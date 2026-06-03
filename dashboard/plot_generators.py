"""Plotly figure generators for the sweep dashboard."""

from typing import List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .utils import format_column_name, get_color_scale


def _empty(msg: str) -> go.Figure:
    return go.Figure().add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False, font=dict(size=14)
    )


def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str,
                        color_by: Optional[str] = None) -> go.Figure:
    if x_col not in df.columns or y_col not in df.columns:
        return _empty("Invalid column selection")
    kwargs = dict(
        x=x_col, y=y_col,
        title=f"{format_column_name(y_col)} vs {format_column_name(x_col)}",
        labels={x_col: format_column_name(x_col), y_col: format_column_name(y_col)},
        hover_data=[c for c in ["name", "source", "blur_type"] if c in df.columns],
    )
    if color_by and color_by in df.columns:
        kwargs["color"] = color_by
    fig = px.scatter(df, **kwargs)
    fig.update_traces(marker=dict(size=7, opacity=0.75))
    fig.update_layout(height=600, hovermode="closest")
    return fig


def create_line_plot(df: pd.DataFrame, x_col: str, y_cols: List[str],
                     color_by: Optional[str] = None) -> go.Figure:
    if x_col not in df.columns:
        return _empty("Invalid x-axis column")
    df_s = df.sort_values(x_col)
    fig = go.Figure()
    if color_by and color_by in df.columns:
        groups = df_s[color_by].unique()
        colors = get_color_scale(len(groups) * len(y_cols))
        idx = 0
        for g in groups:
            dg = df_s[df_s[color_by] == g]
            for y in y_cols:
                if y in dg.columns:
                    fig.add_trace(go.Scatter(
                        x=dg[x_col], y=dg[y], mode="lines+markers",
                        name=f"{format_column_name(y)} ({g})",
                        line=dict(color=colors[idx]), marker=dict(size=5)
                    ))
                    idx += 1
    else:
        colors = get_color_scale(len(y_cols))
        for i, y in enumerate(y_cols):
            if y in df_s.columns:
                fig.add_trace(go.Scatter(
                    x=df_s[x_col], y=df_s[y], mode="lines+markers",
                    name=format_column_name(y),
                    line=dict(color=colors[i]), marker=dict(size=5)
                ))
    fig.update_layout(
        title=f"Metrics vs {format_column_name(x_col)}",
        xaxis_title=format_column_name(x_col), yaxis_title="Value",
        height=600, hovermode="x unified"
    )
    return fig


def create_box_plot(df: pd.DataFrame, category_col: str,
                    metric_col: str) -> go.Figure:
    if category_col not in df.columns or metric_col not in df.columns:
        return _empty("Invalid column selection")
    fig = px.box(
        df, x=category_col, y=metric_col, color=category_col,
        title=f"{format_column_name(metric_col)} by {format_column_name(category_col)}",
        labels={category_col: format_column_name(category_col),
                metric_col: format_column_name(metric_col)},
    )
    fig.update_layout(height=600, showlegend=False)
    return fig


def create_correlation_heatmap(df: pd.DataFrame,
                                columns: Optional[List[str]] = None) -> go.Figure:
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    columns = [c for c in columns if c in df.columns]
    if len(columns) < 2:
        return _empty("Select at least 2 numeric columns")
    corr = df[columns].corr()
    labels = [format_column_name(c) for c in columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont={"size": 10}, colorbar=dict(title="r"),
    ))
    fig.update_layout(
        title="Correlation Matrix", height=700,
        xaxis=dict(tickangle=-45),
    )
    return fig


def create_best_config_table(df: pd.DataFrame,
                              display_cols: List[str]) -> go.Figure:
    display_cols = [c for c in display_cols if c in df.columns]
    if not display_cols:
        return _empty("No columns to display")
    values = []
    for col in display_cols:
        col_data = df[col].tolist()
        if df[col].dtype == float:
            col_data = [f"{v:.4f}" if pd.notna(v) else "—" for v in col_data]
        values.append(col_data)
    fig = go.Figure(go.Table(
        header=dict(
            values=[format_column_name(c) for c in display_cols],
            fill_color="#4C78A8", font=dict(color="white", size=12), align="left"
        ),
        cells=dict(
            values=values,
            fill_color=["#f0f4f8" if i % 2 == 0 else "white"
                        for i in range(len(display_cols))],
            align="left", font=dict(size=11)
        )
    ))
    fig.update_layout(height=max(300, 50 + len(df) * 28))
    return fig


def create_multi_metric_comparison(df: pd.DataFrame, x_col: str,
                                    metric_cols: List[str],
                                    normalize: bool = True) -> go.Figure:
    if x_col not in df.columns:
        return _empty("Invalid x-axis column")
    metric_cols = [c for c in metric_cols if c in df.columns]
    if not metric_cols:
        return _empty("No valid metric columns")
    df_s = df.sort_values(x_col)
    colors = get_color_scale(len(metric_cols))
    fig = go.Figure()
    for i, m in enumerate(metric_cols):
        y = df_s[m].values.astype(float)
        if normalize:
            lo, hi = np.nanmin(y), np.nanmax(y)
            if hi > lo:
                y = (y - lo) / (hi - lo)
        fig.add_trace(go.Scatter(
            x=df_s[x_col], y=y, mode="lines+markers",
            name=format_column_name(m),
            line=dict(color=colors[i]), marker=dict(size=5)
        ))
    title = f"Multi-Metric vs {format_column_name(x_col)}"
    if normalize:
        title += " (Normalized)"
    fig.update_layout(
        title=title,
        xaxis_title=format_column_name(x_col),
        yaxis_title="Normalized" if normalize else "Value",
        height=600, hovermode="x unified",
        legend=dict(orientation="v", x=1.02, y=1, xanchor="left", yanchor="top"),
    )
    return fig
