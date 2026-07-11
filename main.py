import sys
import logging

from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(threadName)s] - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("车牌马赛克工具")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
