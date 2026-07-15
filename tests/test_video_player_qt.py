"""最小播放器测试（QtMultimedia 版）：选择文件 → 加载 → 播放。
仅依赖 PySide6 QtMultimedia，不含 OpenCV。"""

import sys
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 显式将 exe 目录加入 DLL 搜索路径，解决 Nuitka 下 Qt 插件加载问题
_exe_dir = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) or '__compiled__' in dir(__builtins__) else os.path.dirname(__file__)
if os.path.isdir(_exe_dir):
    os.add_dll_directory(_exe_dir)

# 日志：同时写文件和控制台
LOG_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".hide-license")) / "hide-license"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "test_player_qt.log"

root = logging.getLogger()
root.setLevel(logging.INFO)
fmt = logging.Formatter(
    "%(asctime)s - [%(threadName)s] - %(name)s - %(levelname)s - %(message)s",
    "%H:%M:%S",
)
fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)
root.addHandler(fh)
ch = logging.StreamHandler(sys.stderr)
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
root.addHandler(ch)

logger = logging.getLogger("test_player_qt")
logger.info(f"日志文件: {LOG_FILE}")

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer, QUrl, qInstallMessageHandler, QtMsgType
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget


def _qt_msg_handler(msg_type, context, msg):
    names = {
        QtMsgType.QtDebugMsg: "DEBUG",
        QtMsgType.QtInfoMsg: "INFO",
        QtMsgType.QtWarningMsg: "WARNING",
        QtMsgType.QtCriticalMsg: "CRITICAL",
        QtMsgType.QtFatalMsg: "FATAL",
    }
    level = names.get(msg_type, "UNKNOWN")
    logger.info(f"Qt[{level}] {msg}")

qInstallMessageHandler(_qt_msg_handler)

VIDEO_FILTER = "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;所有文件 (*.*)"


def _fmt_time(ms):
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


class VideoPlayerQt(QWidget):
    """基于 QtMultimedia 的视频播放器"""

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self._name = name
        self._seeking = False

        self._player = QMediaPlayer()
        audio = QAudioOutput()
        self._player.setAudioOutput(audio)

        self._video = QVideoWidget()
        self._video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._video.setMinimumSize(320, 240)
        self._player.setVideoOutput(self._video)

        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedWidth(36)
        self._btn_play.clicked.connect(self._toggle_play)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderReleased.connect(self._on_slider_released)

        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setFixedWidth(110)
        self._time_label.setAlignment(Qt.AlignCenter)

        self._status_label = QLabel("等待加载")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._video, stretch=1)

        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(0, 4, 0, 0)
        ctrl.addWidget(self._btn_play)
        ctrl.addWidget(self._slider, stretch=1)
        ctrl.addWidget(self._time_label)
        layout.addLayout(ctrl)
        layout.addWidget(self._status_label)

        # Signals
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._on_state)
        self._player.mediaStatusChanged.connect(self._on_media_status)
        self._player.errorOccurred.connect(self._on_error)

    def load(self, path):
        logger.info(f"[{self._name}] load(): {path}")
        self._player.stop()
        self._status_label.setText(f"正在加载: {Path(path).name}")
        self._player.setSource(QUrl.fromLocalFile(path))
        self._player.play()

    def stop(self):
        logger.info(f"[{self._name}] stop()")
        self._player.stop()

    def reset(self):
        self._player.stop()
        self._player.setSource(QUrl())
        self._slider.setValue(0)
        self._time_label.setText("00:00 / 00:00")
        self._btn_play.setText("▶")
        self._video.hide()
        self._video.show()
        self._status_label.setText("等待加载")

    def _toggle_play(self):
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            logger.info(f"[{self._name}] pause()")
            self._player.pause()
        elif self._player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia:
            self._player.setPosition(0)
            self._player.play()
        else:
            self._player.play()

    def _on_position(self, pos):
        if not self._seeking:
            self._slider.setValue(pos)
        dur = self._player.duration()
        self._time_label.setText(f"{_fmt_time(pos)} / {_fmt_time(dur)}")

    def _on_duration(self, dur):
        logger.info(f"[{self._name}] duration={dur}ms")
        self._slider.setRange(0, dur)
        self._time_label.setText(f"00:00 / {_fmt_time(dur)}")

    def _on_state(self, state):
        names = {QMediaPlayer.StoppedState: "Stopped",
                 QMediaPlayer.PlayingState: "Playing",
                 QMediaPlayer.PausedState: "Paused"}
        logger.info(f"[{self._name}] 播放状态: {names.get(state, state)}")
        if state == QMediaPlayer.PlayingState:
            self._btn_play.setText("⏸")
        else:
            self._btn_play.setText("▶")

    def _on_media_status(self, status):
        names = {0: "NoMedia", 1: "LoadingMedia", 2: "LoadedMedia", 3: "StalledMedia",
                 4: "BufferingMedia", 5: "BufferedMedia", 6: "EndOfMedia", 7: "InvalidMedia"}
        logger.info(f"[{self._name}] 媒体状态: {names.get(status, status)}")
        if status == QMediaPlayer.MediaStatus.LoadedMedia:
            self._status_label.setText("已加载")
        elif status == QMediaPlayer.MediaStatus.InvalidMedia:
            self._status_label.setText("加载失败: 格式不支持")
        elif status == QMediaPlayer.MediaStatus.EndOfMedia:
            self._player.setPosition(0)
            self._player.pause()

    def _on_error(self, err, err_str):
        logger.error(f"[{self._name}] 播放错误: {err} {err_str}")
        self._status_label.setText(f"错误: {err_str}")

    def _on_slider_pressed(self):
        self._seeking = True

    def _on_slider_released(self):
        self._seeking = False
        self._player.setPosition(self._slider.value())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("播放器测试 (QtMultimedia)")
        self.resize(900, 550)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        bar = QWidget()
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(0, 0, 0, 0)
        self._file_edit = QLabel("未选择文件")
        bl.addWidget(self._file_edit, stretch=1)
        btn = QPushButton("选择视频")
        btn.clicked.connect(self._on_browse)
        bl.addWidget(btn)
        root.addWidget(bar)

        self._player = VideoPlayerQt("qt_test")
        root.addWidget(self._player, stretch=1)

        logger.info("MainWindow 初始化完成")

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", VIDEO_FILTER)
        if not path:
            logger.info("用户取消选择")
            return
        self._file_edit.setText(path)
        logger.info(f"用户选择: {path}")
        self._player.load(path)


def main():
    logger.info(f"Python: {sys.version}")
    logger.info(f"Executable: {sys.executable}")
    logger.info("使用 QtMultimedia 后端")

    try:
        app = QApplication(sys.argv)
        app.setApplicationName("播放器测试 QtMultimedia")
        logger.info("QApplication created")
        window = MainWindow()
        window.show()
        logger.info(f"Window visible: {window.isVisible()}")
        logger.info("Entering event loop...")
        rc = app.exec()
        logger.info(f"Event loop exited, rc={rc}")
        sys.exit(rc)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
