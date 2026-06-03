import logging
import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

_DARK = """
QWidget {
    background-color: #1e1e1e;
    color: #d4d4d4;
    font-family: "Segoe UI";
    font-size: 9pt;
}
QLineEdit, QComboBox {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 3px;
    padding: 2px 4px;
    color: #d4d4d4;
}
QSpinBox, QDoubleSpinBox {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 3px;
    padding: 2px 4px;
    color: #d4d4d4;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 16px;
    border-left: 1px solid #3d3d3d;
    border-bottom: 1px solid #3d3d3d;
    background-color: #3a3a3a;
    border-top-right-radius: 2px;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
    background-color: #4a4a4a;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 16px;
    border-left: 1px solid #3d3d3d;
    background-color: #3a3a3a;
    border-bottom-right-radius: 2px;
}
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #4a4a4a;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    width: 8px;
    height: 5px;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    width: 8px;
    height: 5px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background-color: #2d2d2d;
    selection-background-color: #3d3d3d;
}
QPushButton {
    background-color: #3a3a3a;
    border: 1px solid #555;
    border-radius: 3px;
    padding: 4px 10px;
    color: #d4d4d4;
}
QPushButton:hover  { background-color: #4a4a4a; }
QPushButton:pressed { background-color: #2a2a2a; }
QTabWidget::pane   { border: 1px solid #3d3d3d; }
QTabBar::tab {
    background-color: #2d2d2d;
    padding: 6px 16px;
    margin-right: 2px;
    border-top-left-radius: 3px;
    border-top-right-radius: 3px;
}
QTabBar::tab:selected { background-color: #3d3d3d; }
QSplitter::handle   { background-color: #3d3d3d; }
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #666;
    border-radius: 3px;
    background-color: #2d2d2d;
}
QCheckBox::indicator:checked { background-color: #4488FF; }
QMenuBar {
    background-color: #1e1e1e;
    border-bottom: 1px solid #3d3d3d;
}
QMenuBar::item:selected { background-color: #3d3d3d; }
QMenu {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
}
QMenu::item:selected { background-color: #3d3d3d; }
"""


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(_DARK)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
