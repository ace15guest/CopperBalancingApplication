import tempfile
from pathlib import Path

import plotly.graph_objects as go
from PyQt6.QtCore import QUrl
from PyQt6.QtWebEngineWidgets import QWebEngineView
from numpy.typing import NDArray

# Write plotly HTML to a temp file so the embedded browser can load
# local resources. Using "cdn" requires external network access which
# QWebEngineView blocks by default.
_TEMP_DIR = Path(tempfile.gettempdir())
_instance_counter = 0


class HeatmapView(QWebEngineView):
    """Renders a warpage or density heatmap using Plotly in an embedded browser."""

    def __init__(self, parent=None):
        super().__init__(parent)
        global _instance_counter
        _instance_counter += 1
        self._temp_file = _TEMP_DIR / f"copper_heatmap_{_instance_counter}.html"
        self._show_empty()

    def _render(self, fig: go.Figure) -> None:
        import plotly.io as pio
        pio.write_html(
            fig,
            file=str(self._temp_file),
            include_plotlyjs=True,
            full_html=True,
        )
        self.load(QUrl.fromLocalFile(str(self._temp_file)))

    def _show_empty(self):
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#2d2d2d",
            font_color="#d4d4d4",
            margin=dict(l=40, r=20, t=20, b=40),
        )
        self._render(fig)

    def show_heatmap(self, z: NDArray, x: NDArray, y: NDArray, title: str = "") -> None:
        fig = go.Figure(go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale="RdBu_r",
            colorbar=dict(title="mm"),
        ))
        fig.update_layout(
            title=title,
            paper_bgcolor="#1e1e1e",
            plot_bgcolor="#2d2d2d",
            font_color="#d4d4d4",
            margin=dict(l=40, r=20, t=40, b=40),
            xaxis_title="x (mm)",
            yaxis_title="y (mm)",
        )
        self._render(fig)

    def show_surface(self, z: NDArray, x: NDArray, y: NDArray, title: str = "") -> None:
        import numpy as np
        zmin = float(np.nanmin(z))
        zmax = float(np.nanmax(z))
        # Symmetric colorscale so zero displacement is the neutral midpoint
        clim = max(abs(zmin), abs(zmax))
        fig = go.Figure(go.Surface(
            z=z,
            x=x,
            y=y,
            colorscale="RdBu_r",
            cmin=-clim,
            cmax=clim,
            colorbar=dict(title="mm"),
        ))
        fig.update_layout(
            title=title,
            paper_bgcolor="#1e1e1e",
            font_color="#d4d4d4",
            margin=dict(l=0, r=0, t=40, b=0),
            scene=dict(
                xaxis_title="x (mm)",
                yaxis_title="y (mm)",
                zaxis_title="z (mm)",
                xaxis=dict(backgroundcolor="#2d2d2d", gridcolor="#444"),
                yaxis=dict(backgroundcolor="#2d2d2d", gridcolor="#444"),
                zaxis=dict(backgroundcolor="#1e1e1e", gridcolor="#444"),
                # Default camera: straight down so it looks like a heatmap
                camera=dict(
                    eye=dict(x=0, y=0, z=2.5),
                    up=dict(x=0, y=1, z=0),
                ),
            ),
        )
        self._render(fig)
