"""Sweep results analysis dashboard."""

from pathlib import Path
import pandas as pd
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

from .data_loader import DataLoader
from .plot_generators import (
    create_scatter_plot, create_line_plot, create_box_plot,
    create_correlation_heatmap, create_best_config_table,
    create_multi_metric_comparison,
)
from .utils import format_column_name, create_filter_options, get_metric_description

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                suppress_callback_exceptions=True)


def create_layout(dl: DataLoader) -> html.Div:
    param_cols   = dl.get_parameter_columns()
    metric_cols  = dl.get_metric_columns()
    num_params   = dl.get_numeric_parameters()
    cat_params   = dl.get_categorical_parameters()

    # Filter dropdowns — up to 6 categorical columns
    filter_cols = cat_params[:6]
    filter_controls = [
        dbc.Col([
            html.Label(format_column_name(col), className="fw-bold small"),
            dcc.Dropdown(
                id=f"filter-{col}",
                options=create_filter_options(dl.df, col),
                multi=True,
                placeholder=f"All {format_column_name(col)}",
                className="mb-2",
            )
        ], md=2)
        for col in filter_cols
    ]

    return html.Div([
        # Header
        dbc.Container([
            html.H2("Sweep Analysis Dashboard", className="text-center my-3 text-primary"),
            html.Hr(),
        ], fluid=True),

        # Filters
        dbc.Container([
            html.H5("Filters", className="mb-2"),
            dbc.Row(filter_controls),
            html.Hr(),
        ], fluid=True, className="mb-3"),

        # Main plot controls
        dbc.Container([
            html.H5("Main Visualization", className="mb-2"),
            dbc.Row([
                dbc.Col([
                    html.Label("X-Axis", className="fw-bold small"),
                    dcc.Dropdown(
                        id="x-axis", clearable=False,
                        options=[{"label": format_column_name(c), "value": c} for c in param_cols],
                        value=param_cols[0] if param_cols else None,
                    )
                ], md=3),
                dbc.Col([
                    html.Label("Y-Axis (Metric)", className="fw-bold small"),
                    dcc.Dropdown(
                        id="y-axis", clearable=False,
                        options=[{"label": format_column_name(c), "value": c} for c in metric_cols],
                        value=metric_cols[0] if metric_cols else None,
                    )
                ], md=3),
                dbc.Col([
                    html.Label("Color By", className="fw-bold small"),
                    dcc.Dropdown(
                        id="color-by",
                        options=[{"label": format_column_name(c), "value": c} for c in cat_params],
                        value=cat_params[0] if cat_params else None,
                        placeholder="None",
                    )
                ], md=3),
                dbc.Col([
                    html.Label("Plot Type", className="fw-bold small"),
                    dcc.RadioItems(
                        id="plot-type",
                        options=[
                            {"label": " Scatter", "value": "scatter"},
                            {"label": " Line",    "value": "line"},
                            {"label": " Box",     "value": "box"},
                        ],
                        value="scatter", inline=True, className="mt-2",
                    )
                ], md=3),
            ], className="mb-2"),
            html.Div(id="metric-desc", className="text-muted small mb-2"),
            dcc.Loading(dcc.Graph(id="main-plot", style={"height": "550px"})),
            html.Div(id="data-count", className="text-center text-muted small mt-1"),
            html.Hr(),
        ], fluid=True, className="mb-3"),

        # Tabs
        dbc.Container([
            dbc.Tabs([
                dbc.Tab(label="Correlation", children=[
                    html.Div([
                        html.H6("Correlation Heatmap", className="mt-3 mb-2"),
                        dcc.Dropdown(
                            id="corr-cols",
                            options=[{"label": format_column_name(c), "value": c}
                                     for c in num_params + metric_cols],
                            value=(num_params + metric_cols)[:10],
                            multi=True, className="mb-3",
                        ),
                        dcc.Loading(dcc.Graph(id="corr-heatmap", style={"height": "650px"})),
                    ])
                ]),

                dbc.Tab(label="Multi-Metric", children=[
                    html.Div([
                        html.H6("Compare Multiple Metrics", className="mt-3 mb-2"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("X-Axis", className="fw-bold small"),
                                dcc.Dropdown(
                                    id="multi-x",
                                    options=[{"label": format_column_name(c), "value": c}
                                             for c in num_params],
                                    value=num_params[0] if num_params else None,
                                )
                            ], md=4),
                            dbc.Col([
                                html.Label("Metrics", className="fw-bold small"),
                                dcc.Dropdown(
                                    id="multi-metrics",
                                    options=[{"label": format_column_name(c), "value": c}
                                             for c in metric_cols],
                                    value=metric_cols[:3], multi=True,
                                )
                            ], md=6),
                            dbc.Col([
                                html.Label("Normalize", className="fw-bold small"),
                                dcc.Checklist(
                                    id="normalize-cb",
                                    options=[{"label": " 0–1", "value": "yes"}],
                                    value=["yes"], className="mt-2",
                                )
                            ], md=2),
                        ], className="mb-3"),
                        dcc.Loading(dcc.Graph(id="multi-plot", style={"height": "550px"})),
                    ])
                ]),

                dbc.Tab(label="Best Configs", children=[
                    html.Div([
                        html.H6("Find Optimal Combinations", className="mt-3 mb-2"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Metric", className="fw-bold small"),
                                dcc.Dropdown(
                                    id="best-metric", clearable=False,
                                    options=[{"label": format_column_name(c), "value": c}
                                             for c in metric_cols],
                                    value=metric_cols[0] if metric_cols else None,
                                )
                            ], md=4),
                            dbc.Col([
                                html.Label("Objective", className="fw-bold small"),
                                dcc.RadioItems(
                                    id="best-obj",
                                    options=[{"label": " Minimize", "value": "min"},
                                             {"label": " Maximize", "value": "max"}],
                                    value="min", inline=True, className="mt-2",
                                )
                            ], md=4),
                            dbc.Col([
                                html.Label("Top N", className="fw-bold small"),
                                dcc.Slider(
                                    id="top-n", min=5, max=50, step=5, value=10,
                                    marks={i: str(i) for i in range(5, 51, 10)},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                )
                            ], md=4),
                        ], className="mb-3"),
                        dcc.Loading(dcc.Graph(id="best-table", style={"height": "480px"})),
                    ])
                ]),

                dbc.Tab(label="Statistics", children=[
                    html.Div([
                        html.H6("Summary Statistics", className="mt-3 mb-2"),
                        html.Div(id="stats-content"),
                    ])
                ]),
            ])
        ], fluid=True, className="mb-4"),

        dbc.Container([
            html.Hr(),
            html.P("Copper Balancing — Sweep Analysis Dashboard",
                   className="text-center text-muted small"),
        ], fluid=True),
    ])


def register_callbacks(dl: DataLoader):
    cat_params = dl.get_categorical_parameters()[:6]
    filter_inputs = [Input(f"filter-{c}", "value") for c in cat_params]

    def _apply_filters(filter_values):
        filters = {}
        for col, val in zip(cat_params, filter_values):
            if val:
                filters[col] = val
        return dl.filter_data(filters)

    @app.callback(
        Output("main-plot", "figure"),
        Output("data-count", "children"),
        [Input("x-axis", "value"), Input("y-axis", "value"),
         Input("color-by", "value"), Input("plot-type", "value")]
        + filter_inputs,
    )
    def update_main(x, y, color, ptype, *fv):
        df = _apply_filters(fv)
        if ptype == "scatter":
            fig = create_scatter_plot(df, x, y, color)
        elif ptype == "line":
            fig = create_line_plot(df, x, [y], color)
        elif ptype == "box":
            fig = create_box_plot(df, color or x, y)
        else:
            fig = create_scatter_plot(df, x, y, color)
        return fig, f"Showing {len(df):,} of {len(dl.df):,} rows"

    @app.callback(Output("metric-desc", "children"), Input("y-axis", "value"))
    def update_desc(metric):
        return get_metric_description(metric) if metric else ""

    @app.callback(
        Output("corr-heatmap", "figure"),
        [Input("corr-cols", "value")] + filter_inputs,
    )
    def update_corr(cols, *fv):
        return create_correlation_heatmap(_apply_filters(fv), cols)

    @app.callback(
        Output("multi-plot", "figure"),
        [Input("multi-x", "value"), Input("multi-metrics", "value"),
         Input("normalize-cb", "value")] + filter_inputs,
    )
    def update_multi(x, metrics, norm, *fv):
        return create_multi_metric_comparison(
            _apply_filters(fv), x, metrics or [],
            normalize="yes" in (norm or [])
        )

    @app.callback(
        Output("best-table", "figure"),
        [Input("best-metric", "value"), Input("best-obj", "value"),
         Input("top-n", "value")] + filter_inputs,
    )
    def update_best(metric, obj, n, *fv):
        filters = {col: val for col, val in zip(cat_params, fv) if val}
        best = dl.find_best_configs(metric, top_n=n or 10,
                                    minimize=(obj == "min"), filters=filters)
        cols = dl.get_parameter_columns() + [metric]
        return create_best_config_table(best, cols)

    @app.callback(
        Output("stats-content", "children"),
        [Input("y-axis", "value")] + filter_inputs,
    )
    def update_stats(metric, *fv):
        filters = {col: val for col, val in zip(cat_params, fv) if val}
        stats = dl.get_summary_statistics(metric, filters)
        if not stats:
            return html.P("No data available")
        labels = {"count": "Count", "mean": "Mean", "std": "Std Dev",
                  "min": "Min", "q25": "25th %ile", "median": "Median",
                  "q75": "75th %ile", "max": "Max"}
        cards = [
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6(label, className="text-muted small"),
                html.H5(str(stats[k]) if k == "count"
                        else f"{stats[k]:.4f}"),
            ])), md=3, className="mb-3")
            for k, label in labels.items() if k in stats
        ]
        return dbc.Row(cards)


def run_dashboard(data_path: str, host: str = "127.0.0.1",
                  port: int = 8050, debug: bool = False):
    print(f"Loading {data_path} ...")
    dl = DataLoader(data_path)
    app.layout = create_layout(dl)
    register_callbacks(dl)
    print(f"Dashboard at http://{host}:{port}  (Ctrl+C to stop)")
    app.run(host=host, port=port, debug=debug)
