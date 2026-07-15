"""最小播放器测试：选择文件 → 加载 → 播放。仅依赖 OpenCV + PySide6。"""
import sys
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# 日志：同时写文件和控制台，方便排查
LOG_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".hide-license")) / "hide-license"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "test_player.log"

root = logging.getLogger()
root.setLevel(logging.INFO)
fmt = logging.Formatter(
    "%(asctime)s - [%(threadName)s] - %(name)s - %(levelname)s - %(message)s",
    "%H:%M:%S",
)
# 文件 handler
fh = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=2, encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(fmt)
root.addHandler(fh)
# 控制台 handler
ch = logging.StreamHandler(sys.stderr)
ch.setLevel(logging.INFO)
ch.setFormatter(fmt)
root.addHandler(ch)

logger = logging.getLogger("test_player")
logger.info(f"日志文件: {LOG_FILE}")

import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QSizePolicy,
)
from PySide6.QtCore import Qt, QTimer, qInstallMessageHandler, QtMsgType
from PySide6.QtGui import QImage, QPixmap


def _qt_msg_handler(msg_type, context, msg):
    """捕获所有 Qt 内部日志，输出到 Python logging"""
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


class VideoPlayer(QWidget):
    """基于 OpenCV + QLabel 的视频播放器"""

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self._name = name
        self._cap = None
        self._fps = 30.0
        self._total_frames = 0
        self._current_frame = 0
        self._playing = False
        self._seeking = False

        self._video_label = QLabel()
        self._video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._video_label.setMinimumSize(320, 240)
        self._video_label.setAlignment(Qt.AlignCenter)
        self._video_label.setStyleSheet("background-color: black;")
        self._video_label.setText("未加载视频")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)

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
        layout.addWidget(self._video_label, stretch=1)

        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(0, 4, 0, 0)
        ctrl.addWidget(self._btn_play)
        ctrl.addWidget(self._slider, stretch=1)
        ctrl.addWidget(self._time_label)
        layout.addLayout(ctrl)
        layout.addWidget(self._status_label)

    def load(self, path):
        logger.info(f"[{self._name}] load() 开始: {path}")
        self.stop()
        self._status_label.setText(f"正在打开: {Path(path).name}")
        QApplication.processEvents()

        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            logger.error(f"[{self._name}] cv2.VideoCapture 打开失败: {path}")
            self._status_label.setText(f"打开失败: {Path(path).name}")
            self._cap = None
            return

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = 30.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._current_frame = 0
        self._slider.setRange(0, max(0, self._total_frames - 1))

        logger.info(f"[{self._name}] 视频属性: {self._fps:.2f}FPS, {self._total_frames}帧")
        self._status_label.setText(f"已加载: {Path(path).name} ({self._fps:.1f}FPS, {self._total_frames}帧)")

        # 显示第一帧
        ret = self._show_frame(0)
        if not ret:
            logger.error(f"[{self._name}] 读取第一帧失败")
            self._status_label.setText("读取第一帧失败")
        else:
            logger.info(f"[{self._name}] 第一帧显示成功")

    def play(self):
        if not self._cap:
            logger.warning(f"[{self._name}] play() 失败: 未加载视频")
            return
        if self._current_frame >= self._total_frames - 1:
            self._seek_to(0)
        self._playing = True
        self._btn_play.setText("⏸")
        interval = max(1, int(1000.0 / self._fps))
        self._timer.start(interval)
        logger.info(f"[{self._name}] play() 开始, 帧间隔={interval}ms")

    def pause(self):
        self._timer.stop()
        self._playing = False
        self._btn_play.setText("▶")
        logger.info(f"[{self._name}] pause(), 当前帧={self._current_frame}")

    def stop(self):
        self._timer.stop()
        self._playing = False
        if self._cap:
            self._cap.release()
            self._cap = None
        self._btn_play.setText("▶")
        logger.info(f"[{self._name}] stop()")

    def reset(self):
        self.stop()
        self._total_frames = 0
        self._current_frame = 0
        self._slider.setValue(0)
        self._slider.setRange(0, 0)
        self._time_label.setText("00:00 / 00:00")
        self._video_label.setText("未加载视频")
        self._status_label.setText("等待加载")

    def _show_frame(self, frame_idx):
        if not self._cap or frame_idx < 0:
            return False
        if frame_idx != self._current_frame:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self._cap.read()
        if not ret:
            logger.warning(f"[{self._name}] _show_frame({frame_idx}) 读取失败")
            return False
        self._current_frame = frame_idx
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        label_w = self._video_label.width()
        label_h = self._video_label.height()
        if label_w > 10 and label_h > 10:
            scale = min(label_w / w, label_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            if new_w > 0 and new_h > 0:
                rgb = cv2.resize(rgb, (new_w, new_h))
        bytes_per_line = ch * rgb.shape[1]
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                     bytes_per_line, QImage.Format_RGB888)
        self._video_label.setPixmap(QPixmap.fromImage(img))
        self._update_time_label()
        return True

    def _next_frame(self):
        if not self._cap or not self._playing:
            return
        if self._current_frame >= self._total_frames - 1:
            logger.info(f"[{self._name}] 播放到末尾, 总帧={self._total_frames}")
            self.pause()
            self._slider.setValue(self._total_frames - 1)
            return
        if not self._seeking:
            self._slider.setValue(self._current_frame)
        self._show_frame(self._current_frame + 1)

    def _seek_to(self, frame_idx):
        self._show_frame(frame_idx)
        self._slider.setValue(frame_idx)

    def _toggle_play(self):
        if self._playing:
            self.pause()
        else:
            self.play()

    def _update_time_label(self):
        dur_ms = int(self._total_frames / self._fps * 1000) if self._fps > 0 else 0
        pos_ms = int(self._current_frame / self._fps * 1000) if self._fps > 0 else 0
        self._time_label.setText(f"{_fmt_time(pos_ms)} / {_fmt_time(dur_ms)}")

    def _on_slider_pressed(self):
        self._seeking = True

    def _on_slider_released(self):
        self._seeking = False
        self._seek_to(self._slider.value())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("播放器测试")
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

        self._player = VideoPlayer("test")
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
        self._player.play()


def main():
    logger.info(f"Python: {sys.version}")
    logger.info(f"PySide6: 已导入")
    logger.info(f"OpenCV: {cv2.__version__}")
    logger.info(f"Executable: {sys.executable}")
    logger.info(f"Working dir: {os.getcwd()}")

    try:
        app = QApplication(sys.argv)
        app.setApplicationName("播放器测试")
        logger.info("QApplication created")
        window = MainWindow()
        logger.info("MainWindow created, calling show()...")
        window.show()
        logger.info(f"Window visible: {window.isVisible()}, size: {window.size().width()}x{window.size().height()}")
        logger.info("Entering event loop...")
        rc = app.exec()
        logger.info(f"Event loop exited, rc={rc}")
        sys.exit(rc)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
