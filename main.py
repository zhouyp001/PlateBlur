import sys
import logging

from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow
from utils.logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("车牌马赛克工具")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
