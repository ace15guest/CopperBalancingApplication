from pathlib import Path

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMainWindow, QTabWidget, QFileDialog, QMessageBox
from PyQt6.QtGui import QAction

from ui.pages.ingest_page import IngestPage
from ui.pages.stackup_page import StackupPage
from ui.pages.simulate_page import SimulatePage
from ui.pages.density_sim_page import DensitySimPage
from ui.pages.compare_page import ComparePage
from ui.pages.sweep_page import SweepPage
from src.ingestion.stackup import load_stackup, save_stackup


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PCB Warpage Predictor")
        self.setMinimumSize(1280, 800)
        self._current_file: Path | None = None

        self._stackup_page = StackupPage()

        self._simulate_page = SimulatePage()
        self._density_page  = DensitySimPage()
        self._compare_page  = ComparePage()

        tabs = QTabWidget()
        tabs.addTab(IngestPage(), "Ingest")
        tabs.addTab(self._stackup_page, "Stackup")
        tabs.addTab(self._simulate_page, "CLT Simulate")
        tabs.addTab(self._density_page, "Density Simulation")
        tabs.addTab(self._compare_page, "Compare")
        tabs.addTab(SweepPage(), "ΔT Sweep")
        self.setCentralWidget(tabs)

        # Forward CLT result to Compare page
        self._simulate_page.sim_result_ready.connect(
            lambda r: self._compare_page.load_sim_results(clt=r)
        )
        # Forward density result to Compare page
        self._density_page.density_result_ready.connect(
            self._compare_page.load_density_result
        )

        self._build_menu()
        QTimer.singleShot(0, self._auto_run)

    def _auto_run(self) -> None:
        """Fire all three pipelines automatically on startup."""
        self._simulate_page._run()
        self._density_page._run()
        self._compare_page._load()

    def _build_menu(self):
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("File")

        new_action = QAction("New Stackup", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self._new)

        open_action = QAction("Open Stackup...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open)

        save_action = QAction("Save Stackup", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save)

        save_as_action = QAction("Save Stackup As...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self._save_as)

        file_menu.addAction(new_action)
        file_menu.addSeparator()
        file_menu.addAction(open_action)
        file_menu.addAction(save_action)
        file_menu.addAction(save_as_action)
        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Alt+F4")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    # ------------------------------------------------------------------
    # File actions
    # ------------------------------------------------------------------

    def _new(self):
        if not self._confirm_discard():
            return
        self._stackup_page.new_stackup()
        self._current_file = None
        self._update_title()

    def _open(self):
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Stackup", "", "Stackup Files (*.json);;All Files (*)"
        )
        if not path:
            return
        try:
            stackup = load_stackup(Path(path))
            self._stackup_page.load_stackup(stackup)
            self._current_file = Path(path)
            self._update_title()
        except Exception as e:
            QMessageBox.critical(self, "Open Failed", str(e))

    def _save(self):
        if self._current_file:
            self._write(self._current_file)
        else:
            self._save_as()

    def _save_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Stackup As", "", "Stackup Files (*.json);;All Files (*)"
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"
        self._current_file = Path(path)
        self._write(self._current_file)
        self._update_title()

    def _write(self, path: Path):
        try:
            save_stackup(self._stackup_page.get_stackup(), path)
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", str(e))

    def _confirm_discard(self) -> bool:
        if self._current_file is None and self._stackup_page.get_stackup().rows:
            reply = QMessageBox.question(
                self, "Unsaved Stackup",
                "The current stackup has unsaved changes. Discard and continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            return reply == QMessageBox.StandardButton.Yes
        return True

    def _update_title(self):
        name = self._current_file.name if self._current_file else "Unsaved"
        self.setWindowTitle(f"PCB Warpage Predictor — {name}")
