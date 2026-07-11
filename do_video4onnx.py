import os
import cv2
import time
import logging
from ultralytics import YOLO

# --- 性能优化设置 ---
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

# ---------------- 配置日志 ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ---------------- 辅助函数 ----------------
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


# ---------------- 主程序 ----------------
if __name__ == '__main__':
    # --- 配置参数 ---
    video_path = 'tmp4_end5s.mp4'
    output_path = 'tmp4_end5s_result_onnx.mp4'
    model_path = 'best_imgsz_1280.onnx'
    resize4model = 1280
    confidence = 0.15
    skip_frames = 1

    # --- 初始化变量 (移到 try 外面，防止 finally 报错) ---
    cap = None
    out = None
    last_results = None

    # 统计变量初始化
    start_time = time.time()  # 初始化时间，防止除零错误
    frame_count = 0
    inference_count = 0

    try:
        # 1. 加载模型
        logger.info(f"正在加载 ONNX 模型: {model_path}")
        model = YOLO(model_path)

        # 2. 获取视频属性
        logger.info(f"正在读取视频文件: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")

        width_ = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height_ = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_ = cap.get(cv2.CAP_PROP_FPS)
        if fps_ <= 0: fps_ = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 3. 初始化写入器
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps_, (width_, height_))
        if not out.isOpened():
            raise RuntimeError("无法创建视频写入器")

        logger.info(f"开始处理视频 (总帧数: {total_frames})...")

        # 重置开始时间
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            # --- 跳帧逻辑 ---
            should_infer = False
            if skip_frames == 0:
                should_infer = True
            else:
                if (frame_count - 1) % (skip_frames + 1) == 0:
                    should_infer = True

            if should_infer:
                results = model.predict(
                    source=frame,
                    device="cpu",
                    verbose=False,
                    conf=confidence,
                    imgsz=resize4model,
                    half=False,
                    save=False,
                    show=False
                )
                last_results = results[0]
                inference_count += 1

            # --- 绘制 ---
            processed_frame = draw_mosaic(frame, [last_results] if last_results else None)
            out.write(processed_frame)

            # --- 进度显示 ---
            if frame_count % max(1, total_frames // 10) == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                logger.info(f"进度: {frame_count}/{total_frames} | 速度: {current_fps:.2f} FPS")

    except Exception as e:
        logger.error(f"程序发生致命错误: {e}", exc_info=True)
    finally:
        # --- 资源释放 ---
        if 'cap' in locals() and cap and cap.isOpened():
            cap.release()
        if 'out' in locals() and out and out.isOpened():
            out.release()

        # --- 最终总结日志 (安全访问变量) ---
        total_elapsed = time.time() - start_time
        # 防止除以0
        final_fps = frame_count / total_elapsed if total_elapsed > 0 else 0

        # 安全计算跳帧统计
        saved_frames = frame_count - inference_count
        saved_ratio = (saved_frames / frame_count * 100) if frame_count > 0 else 0

        logger.info("=" * 60)
        logger.info("📊 任务处理总结报告")
        logger.info("=" * 60)
        logger.info(f"📁 输入文件: {video_path}")
        logger.info(f"💾 输出文件: {output_path}")
        logger.info("-" * 60)
        logger.info(f"⏱️  总耗时:   {total_elapsed:.2f} 秒")
        logger.info(f"🚀 平均速度: {final_fps:.2f} FPS")
        logger.info("-" * 60)
        logger.info(f"🎞️  总帧数:   {frame_count} 帧")
        logger.info(f"🧠 推理次数: {inference_count} 次")

        if skip_frames > 0:
            logger.info(f"⚡ 跳帧策略: 每 {skip_frames + 1} 帧推理 1 次")
            logger.info(f"📉 节省计算: 跳过 {saved_frames} 帧推理 (约 {saved_ratio:.1f}%)")
        else:
            logger.info("⚡ 跳帧策略: 无 (全帧处理)")

        logger.info("=" * 60)
        logger.info("✅ 视频资源已释放，程序退出。")
        logger.info("=" * 60)