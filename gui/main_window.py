import os
import time
import shutil
import uuid
import logging
import tempfile
import atexit
from pathlib import Path

import psutil

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLineEdit, QFileDialog, QTabWidget, QProgressBar,
    QGroupBox, QGridLayout, QLabel, QStatusBar, QMessageBox,
    QSizePolicy, QApplication, QSplitter, QSlider,
)
from PySide6.QtCore import Qt, QUrl, QTimer, QThread
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget

from utils.processor import VideoProcessor, auto_detect, load_config

logger = logging.getLogger(__name__)

VIDEO_FILTER = "视频文件 (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm);;所有文件 (*.*)"


def _fmt_time(ms):
    """毫秒 → mm:ss"""
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


class _VideoPlayer(QWidget):
    """带播放控件的视频播放器（QtMultimedia）"""

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

        # Controls
        self._btn_play = QPushButton("▶")
        self._btn_play.setFixedWidth(36)
        self._btn_play.clicked.connect(self._toggle_play)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.sliderPressed.connect(self._on_slider_pressed)
        self._slider.sliderMoved.connect(self._on_slider_moved)
        self._slider.sliderReleased.connect(self._on_slider_released)

        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setFixedWidth(110)
        self._time_label.setAlignment(Qt.AlignCenter)

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._video, stretch=1)

        ctrl = QHBoxLayout()
        ctrl.setContentsMargins(0, 4, 0, 0)
        ctrl.addWidget(self._btn_play)
        ctrl.addWidget(self._slider, stretch=1)
        ctrl.addWidget(self._time_label)
        layout.addLayout(ctrl)

        # Signals
        self._player.positionChanged.connect(self._on_position)
        self._player.durationChanged.connect(self._on_duration)
        self._player.playbackStateChanged.connect(self._on_state)
        self._player.mediaStatusChanged.connect(self._on_media_status)
        self._player.errorOccurred.connect(
            lambda err, es, n=name: logger.error(f"播放器[{n}] 错误: {err} {es}")
        )

    # -- Public API --

    def load(self, path):
        logger.info(f"播放器[{self._name}] 加载: {path}")
        self._player.setSource(QUrl.fromLocalFile(path))

    def preview(self):
        """等待媒体加载完成后播放一帧，然后暂停以显示首帧画面。"""
        def _show_frame(status):
            if status == QMediaPlayer.MediaStatus.LoadedMedia:
                try:
                    self._player.mediaStatusChanged.disconnect(_show_frame)
                except Exception:
                    pass
                logger.info(f"播放器[{self._name}] 媒体已加载，显示首帧")
                self._player.play()
                QTimer.singleShot(500, self._player.pause)

        if self._player.mediaStatus() == QMediaPlayer.MediaStatus.LoadedMedia:
            self._player.play()
            QTimer.singleShot(500, self._player.pause)
        else:
            self._player.mediaStatusChanged.connect(_show_frame)

    def stop(self):
        self._player.stop()

    def play(self):
        if self._player.mediaStatus() == QMediaPlayer.MediaStatus.EndOfMedia:
            self._player.setPosition(0)
        self._player.play()

    def pause(self):
        self._player.pause()

    def reset(self):
        self._player.stop()
        self._player.setSource(QUrl())
        self._slider.setValue(0)
        self._time_label.setText("00:00 / 00:00")
        self._btn_play.setText("▶")

    # -- Slots --

    def _toggle_play(self):
        if self._player.playbackState() == QMediaPlayer.PlayingState:
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
        self._slider.setRange(0, dur)
        self._time_label.setText(f"00:00 / {_fmt_time(dur)}")

    def _on_state(self, state):
        if state == QMediaPlayer.PlayingState:
            self._btn_play.setText("⏸")
        else:
            self._btn_play.setText("▶")

    def _on_media_status(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self._player.setPosition(0)
            self._player.pause()
            self._slider.setValue(0)
            self._btn_play.setText("▶")

    def _on_slider_pressed(self):
        self._seeking = True

    def _on_slider_moved(self, pos):
        self._time_label.setText(f"{_fmt_time(pos)} / {_fmt_time(self._player.duration())}")

    def _on_slider_released(self):
        self._seeking = False
        self._player.setPosition(self._slider.value())


class _ProcessingStats(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("处理统计", parent)
        self._labels = {}
        layout = QGridLayout(self)
        fields = [
            ("进度", "等待开始"), ("总帧数", "-"), ("已处理", "-"),
            ("推理次数", "-"), ("当前速度", "-"), ("跳帧节省", "-"),
            ("预估剩余", "-"), ("运行设备", "-"), ("总耗时", "-"), ("输出文件", "-"),
        ]
        for i, (name, value) in enumerate(fields):
            layout.addWidget(QLabel(f"{name}:"), i, 0)
            val = QLabel(value)
            val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            val.setWordWrap(True)
            layout.addWidget(val, i, 1)
            self._labels[name] = val

    def set_field(self, name, value):
        if name in self._labels:
            self._labels[name].setText(str(value))

    def reset(self, device_info=""):
        for k, v in {
            "进度": "等待开始", "总帧数": "-", "已处理": "-",
            "推理次数": "-", "当前速度": "-", "跳帧节省": "-",
            "预估剩余": "-", "运行设备": device_info, "总耗时": "-", "输出文件": "-",
        }.items():
            self._labels[k].setText(v)

    def set_processing_mode(self):
        """进入处理中状态，初始化运行时字段"""
        self.set_field("预估剩余", "计算中...")
        self.set_field("总耗时", "-")
        self.set_field("输出文件", "-")

    def set_done_mode(self):
        """进入完成状态"""
        self.set_field("预估剩余", "-")


class _ResourceMonitor(QGroupBox):
    def __init__(self, parent=None):
        super().__init__("系统状态", parent)
        layout = QGridLayout(self)

        self._cpu_label = QLabel("CPU: --")
        self._ram_label = QLabel("RAM: --")
        self._gpu_label = QLabel("GPU: --")

        layout.addWidget(QLabel("CPU"), 0, 0)
        layout.addWidget(self._cpu_label, 0, 1)
        layout.addWidget(QLabel("RAM"), 1, 0)
        layout.addWidget(self._ram_label, 1, 1)
        layout.addWidget(QLabel("GPU"), 2, 0)
        layout.addWidget(self._gpu_label, 2, 1)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(2000)
        self._refresh()

    def _refresh(self):
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        self._cpu_label.setText(f"{cpu:.1f}%")
        self._ram_label.setText(f"{mem.percent:.1f}%  ({mem.used // 1024**2} / {mem.total // 1024**2} MB)")
        try:
            import torch
            if torch.cuda.is_available():
                g = torch.cuda.memory_allocated(0) / 1024**3
                gt = torch.cuda.get_device_properties(0).total_memory / 1024**3
                p = (g / gt * 100) if gt > 0 else 0
                self._gpu_label.setText(f"{p:.1f}%  ({g:.1f}/{gt:.1f} GB)")
            else:
                self._gpu_label.setText("-")
        except Exception:
            self._gpu_label.setText("-")

    def stop(self):
        self._timer.stop()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("车牌马赛克工具")
        self.setMinimumSize(1100, 600)
        self.resize(1300, 750)

        self._input_path = None
        self._temp_result_path = None
        self._result_saved = False
        self._running = False
        self._start_time = None
        self._inference_count = 0
        self._last_progress_frame = 0
        self._last_progress_time = None
        self._worker = None
        self._thread = None
        self._tmp_files = set()
        atexit.register(self._cleanup_temp_files)

        self._config = load_config()
        self._device, self._model_path = auto_detect(self._config)
        self._device_info = f"{self._device.upper()} / {Path(self._model_path).name}"
        logger.info(f"MainWindow 初始化: {self._device_info}")

        self._setup_ui()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        root.addWidget(self._make_file_bar())

        splitter = QSplitter(Qt.Horizontal)

        # Left: video tabs with playback controls
        video_panel = QWidget()
        video_layout = QVBoxLayout(video_panel)
        video_layout.setContentsMargins(0, 0, 0, 0)

        self._tabs = QTabWidget()
        self._original = _VideoPlayer("original")
        self._result = _VideoPlayer("result")
        self._tabs.addTab(self._original, "原始视频")
        self._tabs.addTab(self._result, "处理结果")
        video_layout.addWidget(self._tabs)
        splitter.addWidget(video_panel)

        # Right: stats + resource + progress + buttons
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(4, 0, 0, 0)

        self._stats = _ProcessingStats()
        self._stats.reset(self._device_info)
        self._stats.set_field("运行设备", self._device_info)
        right_layout.addWidget(self._stats)

        self._resource_monitor = _ResourceMonitor()
        right_layout.addWidget(self._resource_monitor)

        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        right_layout.addWidget(self._progress_bar)

        right_layout.addWidget(self._make_button_bar())
        right_layout.addStretch()

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage(self._device_info)
        self._update_button_states()

    def _make_file_bar(self):
        bar = QWidget()
        l = QHBoxLayout(bar); l.setContentsMargins(0, 0, 0, 0)
        self._file_edit = QLineEdit()
        self._file_edit.setReadOnly(True)
        self._file_edit.setPlaceholderText("请选择待处理的视频文件...")
        l.addWidget(self._file_edit)
        b = QPushButton("选择视频"); b.clicked.connect(self._on_browse); l.addWidget(b)
        self._btn_new_file = QPushButton("选择新文件")
        self._btn_new_file.clicked.connect(lambda: self._on_browse(is_new=True)); l.addWidget(self._btn_new_file)
        return bar

    def _make_button_bar(self):
        bar = QWidget()
        l = QHBoxLayout(bar); l.setContentsMargins(0, 0, 0, 0)
        self._btn_start = QPushButton("开始处理"); self._btn_start.clicked.connect(self._on_start); l.addWidget(self._btn_start)
        self._btn_cancel = QPushButton("取消"); self._btn_cancel.clicked.connect(self._on_cancel); self._btn_cancel.setVisible(False); l.addWidget(self._btn_cancel)
        l.addStretch()
        self._btn_save = QPushButton("另存为..."); self._btn_save.clicked.connect(self._on_save_as); l.addWidget(self._btn_save)
        self._btn_discard = QPushButton("丢弃结果"); self._btn_discard.clicked.connect(self._on_discard); l.addWidget(self._btn_discard)
        return bar

    def _update_button_states(self):
        has_input = bool(self._input_path)
        has_result = bool(self._temp_result_path and os.path.exists(self._temp_result_path))
        if self._running:
            self._btn_start.setVisible(False); self._btn_cancel.setVisible(True)
            self._btn_save.setEnabled(False); self._btn_discard.setEnabled(False)
        else:
            self._btn_start.setVisible(True); self._btn_cancel.setVisible(False)
            self._btn_start.setEnabled(has_input and not has_result)
            self._btn_save.setEnabled(has_result); self._btn_discard.setEnabled(has_result)
        self._btn_new_file.setEnabled(not self._running)

    def _reset_ui_for_new_file(self):
        self._stats.reset(self._device_info)
        self._progress_bar.setValue(0); self._progress_bar.setVisible(False)
        self._result.reset()
        self._tabs.setCurrentIndex(0)
        self._result_saved = False
        self._inference_count = 0
        self._last_progress_frame = 0
        self._last_progress_time = None

    def _on_browse(self, is_new=False):
        if is_new and self._temp_result_path and os.path.exists(self._temp_result_path) and not self._result_saved:
            action = self._prompt_unsaved()
            if action == "cancel": return
            elif action == "save": self._on_save_as()
            elif action == "discard": self._on_discard()
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", VIDEO_FILTER)
        if not path: return
        self._input_path = path; self._file_edit.setText(path)
        logger.info(f"已选择: {path}")
        self._cleanup_old_temp()
        self._reset_ui_for_new_file()
        self._update_button_states()
        self._original.load(path)
        self._original.play()
        self._tabs.setCurrentIndex(0)

    def _on_start(self):
        if not self._input_path: return
        logger.info("===== 开始处理 =====")
        self._result_saved = False
        self._cleanup_old_temp()
        tmp_name = f"hide_license_{uuid.uuid4().hex[:8]}.mp4"
        self._temp_result_path = os.path.join(tempfile.gettempdir(), tmp_name)
        self._tmp_files.add(self._temp_result_path)
        logger.info(f"临时输出: {self._temp_result_path}")

        self._stats.set_processing_mode()
        self._stats.set_field("运行设备", self._device_info)
        self._progress_bar.setVisible(True); self._progress_bar.setValue(0)
        self._status_bar.showMessage("处理中...")
        self._running = True; self._inference_count = 0
        self._start_time = time.time()
        self._last_progress_frame = 0; self._last_progress_time = self._start_time
        self._update_button_states()

        self._worker = VideoProcessor()
        self._thread = QThread(); self._thread.setObjectName("VideoWorker")
        self._worker.moveToThread(self._thread)
        self._worker.setup(self._input_path, self._temp_result_path,
                           self._config, self._device, self._model_path)
        self._worker.progress.connect(self._on_progress, Qt.QueuedConnection)
        self._worker.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._worker.error.connect(self._on_error, Qt.QueuedConnection)
        self._thread.started.connect(self._worker.run)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()
        logger.info(f"工作线程已启动: {self._thread.objectName()}")

    def _on_cancel(self):
        logger.info("用户取消")
        if self._worker: self._worker.cancel()
        self._running = False; self._update_button_states()
        self._status_bar.showMessage("已取消")

    def _on_progress(self, current, total):
        now = time.time()
        pct = int(current / total * 100) if total > 0 else 0
        self._progress_bar.setValue(pct)
        elapsed = now - self._start_time
        avg_fps = current / elapsed if elapsed > 0 else 0
        skip = self._config.get('skip', 1)
        self._inference_count = (current + skip) // (skip + 1) if skip > 0 else current
        saved = current - self._inference_count
        ratio = (saved / current * 100) if current > 0 else 0
        eta = (total - current) / avg_fps if avg_fps > 0 and total > current else 0
        eta_s = self._format_seconds(eta) if eta > 0 else "计算中..."

        self._stats.set_field("进度", f"{pct}%")
        self._stats.set_field("总帧数", str(total))
        self._stats.set_field("已处理", f"{current} / {total}")
        self._stats.set_field("推理次数", str(self._inference_count))
        self._stats.set_field("当前速度", f"{avg_fps:.1f} FPS")
        self._stats.set_field("跳帧节省", f"{ratio:.0f}% ({saved} 帧)")
        self._stats.set_field("预估剩余", eta_s)
        self._last_progress_frame = current
        self._last_progress_time = now

    def _on_finished(self, output_path):
        logger.info(f"===== 处理完成: {output_path} =====")
        self._running = False
        elapsed = time.time() - self._start_time
        total = self._last_progress_frame
        avg_fps = total / elapsed if elapsed > 0 else 0

        self._stats.set_done_mode()
        self._stats.set_field("进度", "100% ✓")
        self._stats.set_field("总耗时", self._format_seconds(elapsed))
        self._stats.set_field("当前速度", f"{avg_fps:.1f} FPS (平均)")
        self._stats.set_field("预估剩余", "-")
        self._stats.set_field("输出文件", output_path)
        self._progress_bar.setValue(100)
        self._update_button_states()

        # 先清理工作线程，再加载结果视频，避免竞态崩溃
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
            self._thread = None
            self._worker = None

        if os.path.exists(output_path):
            logger.info(f"加载结果视频: {output_path}")
            self._result.load(output_path)
            self._result.preview()
            self._tabs.setCurrentIndex(1)
            self._status_bar.showMessage("处理完成")
        else:
            logger.error(f"输出文件不存在: {output_path}")
            self._status_bar.showMessage("处理完成但输出文件不存在")

    def _on_error(self, message):
        logger.error(f"===== 处理出错: {message} =====")
        self._running = False
        self._progress_bar.setVisible(False)
        self._stats.set_field("进度", f"错误: {message}")
        self._update_button_states()
        QMessageBox.critical(self, "处理出错", f"视频处理失败:\n{message}")
        self._status_bar.showMessage(f"错误: {message}")
        if self._thread:
            self._thread.quit(); self._thread = None; self._worker = None

    def _on_save_as(self):
        if not self._temp_result_path or not os.path.exists(self._temp_result_path): return
        default_name = Path(self._input_path).stem + "_result.mp4" if self._input_path else "result.mp4"
        target, _ = QFileDialog.getSaveFileName(self, "另存为", default_name, "MP4 视频 (*.mp4);;所有文件 (*.*)")
        if target:
            shutil.copy(self._temp_result_path, target)
            self._result_saved = True
            self._status_bar.showMessage(f"已保存到: {target}")
            logger.info(f"已保存: {target}")

    def _on_discard(self):
        logger.info("丢弃结果")
        self._cleanup_old_temp()
        self._result.reset()
        self._temp_result_path = None; self._result_saved = False
        self._reset_ui_for_new_file()
        self._update_button_states()
        self._status_bar.showMessage("结果已丢弃")

    def _prompt_unsaved(self):
        box = QMessageBox(self)
        box.setWindowTitle("未保存结果")
        box.setText("当前处理结果尚未保存，是否先保存？")
        save_btn = box.addButton("保存", QMessageBox.YesRole)
        discard_btn = box.addButton("不保存", QMessageBox.DestructiveRole)
        cancel_btn = box.addButton("取消", QMessageBox.RejectRole)
        box.setDefaultButton(save_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked == save_btn: return "save"
        elif clicked == discard_btn: return "discard"
        return "cancel"

    def _cleanup_old_temp(self):
        if self._temp_result_path:
            try:
                if os.path.exists(self._temp_result_path):
                    os.remove(self._temp_result_path)
            except OSError: pass
            self._tmp_files.discard(self._temp_result_path)
            self._temp_result_path = None

    def _cleanup_temp_files(self):
        for f in list(self._tmp_files):
            try:
                if os.path.exists(f): os.remove(f)
            except OSError: pass
        self._tmp_files.clear()

    def closeEvent(self, event):
        logger.info("关闭窗口")
        if self._temp_result_path and os.path.exists(self._temp_result_path) and not self._result_saved:
            action = self._prompt_unsaved()
            if action == "cancel": event.ignore(); return
            elif action == "save": self._on_save_as()
        if self._worker: self._worker.cancel()
        if self._thread and self._thread.isRunning():
            self._thread.quit(); self._thread.wait(2000)
        self._cleanup_temp_files()
        self._resource_monitor.stop()
        super().closeEvent(event)

    @staticmethod
    def _format_seconds(seconds):
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0: return f"{h}时{m}分{s}秒"
        if m > 0: return f"{m}分{s}秒"
        return f"{s}秒"
