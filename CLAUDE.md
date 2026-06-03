# Copper Balancing Application

## Project Overview

<!-- What this application does and why it exists -->

## Architecture

<!-- High-level description of the system design, major components, and how they interact -->

## Tech Stack

<!-- Languages, frameworks, libraries, and tools used -->

## Project Structure

```
CopperBalancingApplication/
├── main.py                        # Entry point — launches PyQt6 app
├── requirements.txt
├── src/                           # Pure Python, no UI dependencies
│   ├── models.py                  # Shared dataclasses (Stackup, SimResult, etc.)
│   ├── ingestion/
│   │   ├── gerber_parser.py       # Calls gerbv to render Gerber → PNG
│   │   ├── stackup.py             # Stackup load/save
│   │   └── akrometrix.py          # Akrometrix .dat → MeasurementData
│   ├── processing/
│   │   ├── rasterizer.py          # PNG → copper density map (numpy array)
│   │   └── grid.py                # Common spatial grid + point cloud interpolation
│   ├── simulation/
│   │   ├── clt_solver.py          # Classical Lamination Theory (fast)
│   │   └── hifi_solver.py         # High-fidelity solver
│   └── analysis/
│       ├── alignment.py           # Spatial registration of sim to measurement
│       └── metrics.py             # RMS, R², Pearson, hotspot overlap, IPC bow/span
├── ui/                            # PyQt6 frontend
│   ├── main_window.py             # Top-level window with tab navigation
│   ├── pages/
│   │   ├── ingest_page.py         # File loading and stackup definition
│   │   ├── simulate_page.py       # Solver controls and progress
│   │   └── compare_page.py        # Side-by-side results and metrics
│   └── components/
│       ├── heatmap_view.py        # Plotly heatmap in QWebEngineView
│       ├── stackup_editor.py      # Layer-by-layer stackup UI
│       ├── comparison_table.py    # Metrics table (CLT vs hifi vs measured)
│       └── file_dropzone.py       # Drag-and-drop file picker
├── tests/
│   ├── unit/                      # Per-module tests
│   ├── integration/               # Full pipeline on sample boards
│   └── fixtures/                  # Sample Gerber files and known-good stackups
├── data/
│   └── materials.json             # Material property database (E, ν, CTE, Tg)
└── docs/
    └── physics.md                 # CLT derivation, assumptions, IPC references
```

## Development Setup

### Prerequisites

<!-- Required tools and versions -->

### Installation

<!-- Steps to get the project running locally -->

### Running the App

<!-- Commands to start the development server / run the application -->

## Commands

<!-- Frequently used commands for building, testing, linting, etc. -->

| Command | Description |
|---------|-------------|
|         |             |

## Testing

- Keep tests minimal — only write them when needed, not speculatively.
- All tests must live in the `tests/` folder, mirroring the source structure:
  - Unit tests for `src/` modules → `tests/unit/`
  - Full pipeline tests → `tests/integration/`
  - Sample Gerber files, stackup JSON, and other test data → `tests/fixtures/`
- Never place test files or test scripts in the project root or alongside source files.
- Run tests from the project root: `pytest tests/`

## Key Concepts / Domain Knowledge

<!-- Domain-specific terms, business logic, or non-obvious concepts Claude should understand -->

## Conventions

<!-- Coding style, naming conventions, patterns to follow or avoid -->

## External Integrations

<!-- APIs, services, databases, or third-party tools this app depends on -->

## Environment Variables

<!-- Required environment variables and what they control -->

| Variable | Description | Required |
|----------|-------------|----------|
|          |             |          |
