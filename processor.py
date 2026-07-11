import os
import cv2
import time
import yaml
import logging
from pathlib import Path

import torch
from ultralytics import YOLO
from PySide6.QtCore import QObject, QThread, Signal

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = {
    'conf': 0.15,
    'resize': 1280,
    'skip': 1,
    'model_pt': 'best.pt',
    'model_onnx': 'best_imgsz_1280.onnx',
}


def load_config():
    config_path = BASE_DIR / 'config.yml'
    config = dict(DEFAULT_CONFIG)
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            if data:
                for k in DEFAULT_CONFIG:
                    if k in data and data[k] is not None:
                        config[k] = data[k]
            logger.info(f"config.yml 加载成功: conf={config['conf']}, resize={config['resize']}, skip={config['skip']}")
        except Exception as e:
            logger.warning(f"读取 config.yml 失败，使用默认值: {e}")
    else:
        logger.info("config.yml 不存在，使用默认参数")
    if not (0.0 <= config['conf'] <= 1.0):
        logger.warning(f"conf={config['conf']} 超出范围，使用默认值 0.15")
        config['conf'] = 0.15
    return config


def auto_detect(config):
    if torch.cuda.is_available():
        device = "cuda"
        model_path = config['model_pt']
        logger.info(f"检测到 CUDA，使用: {device} / {model_path}")
    else:
        device = "cpu"
        model_path = config['model_onnx']
        logger.info(f"未检测到 CUDA，使用: {device} / {model_path}")

    if not Path(model_path).is_absolute():
        model_path = str(BASE_DIR / model_path)
    return device, model_path


def draw_mosaic(frame, results):
    if not results or len(results) == 0:
        return frame
    frame_copy = frame.copy()
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return frame_copy
    boxes = res.boxes.xyxy.cpu().numpy()
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        h, w = frame_copy.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = frame_copy[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
        frame_copy[y1:y2, x1:x2] = blurred_roi
    return frame_copy


def check_avc1():
    import tempfile
    test_path = os.path.join(tempfile.gettempdir(), "_test_avc1.mp4")
    try:
        writer = cv2.VideoWriter(test_path, cv2.VideoWriter_fourcc(*'avc1'), 30, (64, 64))
        if writer.isOpened():
            writer.release()
            try:
                os.remove(test_path)
            except OSError:
                pass
            logger.info("avc1 (H.264) fourcc 可用")
            return True
        logger.warning("avc1 writer.isOpened() 返回 False")
        return False
    except Exception as e:
        logger.warning(f"avc1 检测失败: {e}")
        return False


class VideoProcessor(QObject):
    """在 QThread 中执行视频处理的工作对象。由 MainWindow 创建和管理线程。"""

    progress = Signal(int, int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False
        self._input = None
        self._output = None
        self._config = None
        self._device = None
        self._model_path = None

    def setup(self, input_path, output_path, config, device, model_path):
        """在 start 之前设置参数。调用于主线程。"""
        self._input = input_path
        self._output = output_path
        self._config = config
        self._device = device
        self._model_path = model_path

    def cancel(self):
        logger.info("VideoProcessor.cancel() 被调用")
        self._cancelled = True

    def run(self):
        """在 QThread 中运行的处理主逻辑。"""
        logger.info(f"===== run() 开始, thread={QThread.currentThread()} =====")
        input_path = self._input
        output_path = self._output
        config = self._config
        device = self._device
        model_path = self._model_path
        cap = None
        out = None
        last_results = None
        start_time = time.time()
        frame_count = 0
        inference_count = 0
        has_error = False

        try:
            logger.info(f"设备: {device}, 模型: {model_path}")

            logger.info(f"加载模型: {model_path}")
            model = YOLO(model_path)
            logger.info("模型加载完成")

            logger.info(f"打开视频: {input_path}")
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频: {input_path}")
                self.error.emit(f"无法打开视频文件: {input_path}")
                has_error = True
                return

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"视频属性: {width}x{height}, {fps:.2f}FPS, {total_frames}帧")

            use_avc1 = check_avc1()
            if use_avc1:
                fourcc = cv2.VideoWriter_fourcc(*'avc1')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            logger.info(f"创建输出: {output_path}")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                logger.error("无法创建输出文件")
                self.error.emit("无法创建视频输出文件")
                has_error = True
                return

            logger.info(f"开始逐帧处理 (skip={config['skip']})...")
            while True:
                if self._cancelled:
                    logger.info("已取消")
                    break

                ret, frame = cap.read()
                if not ret:
                    logger.info(f"读取完毕 ({frame_count} 帧)")
                    break

                frame_count += 1
                skip = config['skip']

                if skip == 0 or (frame_count - 1) % (skip + 1) == 0:
                    results = model.predict(
                        source=frame,
                        device=device,
                        verbose=False,
                        conf=config['conf'],
                        imgsz=config['resize'],
                        save=False,
                        show=False,
                        augment=False,
                        half=False,
                    )
                    last_results = results[0]
                    inference_count += 1

                processed_frame = draw_mosaic(frame, [last_results] if last_results else None)
                out.write(processed_frame)
                self.progress.emit(frame_count, total_frames)

            total_elapsed = time.time() - start_time
            final_fps = frame_count / total_elapsed if total_elapsed > 0 else 0
            logger.info(f"处理完成: {frame_count}帧/{inference_count}推理, "
                        f"{total_elapsed:.1f}s, {final_fps:.2f}FPS")

        except Exception as e:
            logger.error(f"处理异常: {e}", exc_info=True)
            self.error.emit(str(e))
            has_error = True
        finally:
            logger.info("释放资源...")
            if cap is not None and cap.isOpened():
                cap.release()
            if out is not None and out.isOpened():
                out.release()
                logger.info("视频文件已关闭")
            # Only emit finished if no error occurred AND output was produced
            if not has_error and frame_count > 0:
                self.finished.emit(output_path)
