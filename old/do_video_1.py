import os
import cv2
import time
import logging
from ultralytics import YOLO
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

# ---------------- 配置日志 ----------------
# 设置日志格式：时间 - 级别 - 消息
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ---------------- 辅助函数 ----------------
def get_video_props(video_path):
    logger.info(f"正在读取视频文件: {video_path}")
    cap_ = cv2.VideoCapture(video_path)

    if not cap_.isOpened():
        logger.error(f"无法打开视频文件: {video_path}，请检查路径是否正确。")
        raise FileNotFoundError(f"Video file not found or cannot be opened: {video_path}")

    width_ = int(cap_.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_ = int(cap_.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_ = cap_.get(cv2.CAP_PROP_FPS)

    # 处理 FPS 为 0 或无效的情况
    if fps_ <= 0:
        logger.warning("检测到视频 FPS 为 0 或无效，将默认设置为 30.0")
        fps_ = 30.0

    size_ = (width_, height_)
    size_str_ = f"{width_}x{height_}"

    total_frames = int(cap_.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"视频属性 -> 分辨率: {size_str_}, FPS: {fps_:.2f}, 总帧数: {total_frames}")
    return cap_, size_, size_str_, fps_, width_, height_, total_frames

def draw_mosaic(frame, results):
    res = results[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            # --- 安全校验：防止坐标越界 ---
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # --- 核心步骤：提取区域并模糊 ---
            # 1. 截取 ROI (Region of Interest)
            # 注意：OpenCV 切片顺序是 [y:y2, x:x2]
            roi = frame[y1:y2, x1:x2]
            blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
            frame[y1:y2, x1:x2] = blurred_roi
    return frame


# ---------------- 主程序 ----------------
if __name__ == '__main__':
    video_path = 'tmp4_end5s.mp4'
    output_path = 'tmp4_end5s_result1.mp4'
    model_path = '../weights/best.pt'
    resize4model = 1920
    confidence = 0.15

    # 跳帧间隔设置
    # skip_frames = 0: 每帧都处理 (0 跳过 0 帧，即 1, 1, 1...)
    # skip_frames = 1: 处理 1 帧，跳过 1 帧 (每 2 帧处理一次)
    # skip_frames = 3: 处理 1 帧，跳过 3 帧 (每 4 帧处理一次)
    skip_frames = 0
    # ===========================================

    try:
        # 1. 加载模型
        logger.info(f"正在加载 YOLO 模型: {model_path}")
        start_load_time = time.time()
        model = YOLO(model_path)
        load_time = time.time() - start_load_time
        logger.info(f"模型加载完成，耗时: {load_time:.2f} 秒")

        # 2. 获取视频属性
        cap, size, size_str, fps, width, height, total_frames = get_video_props(video_path)

        # 3. 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        logger.info(f"正在初始化视频写入器: {output_path} ({size_str}, {fps:.2f} FPS)")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            logger.error("无法创建视频写入器，检查编解码器或输出路径权限。")
            cap.release()
            exit(1)

        # 4. 开始处理循环
        logger.info("开始处理视频帧...")
        if skip_frames > 0:
            logger.info(f"已启用跳帧模式：每处理 1 帧，跳过 {skip_frames} 帧 (实际每 {skip_frames + 1} 帧推理一次)")
        else:
            logger.info("全帧处理模式：每一帧都进行推理")

        start_time = time.time()
        frame_count = 0

        # --- 跳帧逻辑所需变量 ---
        last_results = None  # 存储上一次的结果对象 (可选，如果只需要图像可不要)

        log_interval = max(1, total_frames // 20)

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("视频读取结束。")
                break

            frame_count += 1

            # --- 核心跳帧逻辑 ---
            # 计算当前是第几个周期 (从 0 开始计数)
            # 如果 skip_frames=0: (0)%1==0, (1)%1==0 ... 每帧都处理
            # 如果 skip_frames=1: (0)%2==0 (处理), (1)%2!=0 (跳过), (2)%2==0 (处理)...
            should_infer = ((frame_count - 1) % (skip_frames + 1) == 0)

            if should_infer:
                # 执行推理
                results = model.predict(
                    source=frame,
                    device="cpu",
                    verbose=False,
                    conf=confidence,
                    imgsz=resize4model,
                    # 一下的性能优化
                    save=False,      # 确保不保存
                    show=False,      # 确保不弹窗 (弹窗在 Linux 无头模式下会报错或阻塞)
                    augment=False,   # 确保关闭 TTA (TTA 会让速度变慢 N 倍)
                    half=False       # CPU 上通常 half=False 更快，除非支持 AVX512_FP16
                )

                # 绘制结果
                # plot_frame = results[0].plot(line_width=2, labels=True, boxes=True, font_size=0.5, conf=True)
                # 更新缓存
                last_results = results
            # else:
            #     plot_frame = last_results[0].plot(img=frame, line_width=2, labels=True, boxes=True, font_size=0.5, conf=True)

            plot_frame = draw_mosaic(frame, last_results)
            # 写入帧
            out.write(plot_frame)

            # --- 进度日志 ---
            if frame_count % log_interval == 0 or frame_count == total_frames:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                progress = (frame_count / total_frames) * 100

                # 日志中显示实际推理次数
                inferred_count = (frame_count + skip_frames) // (skip_frames + 1) if skip_frames > 0 else frame_count

                logger.info(
                    f"进度: {progress:.1f}% ({frame_count}/{total_frames}), "
                    f"已推理: {inferred_count} 帧, "
                    f"当前处理速度: {current_fps:.2f} FPS")

        # 5. 收尾工作
        out.release()
        cap.release()

        total_elapsed = time.time() - start_time
        final_fps = frame_count / total_elapsed if total_elapsed > 0 else 0

        logger.info("=" * 30)
        logger.info("处理完成!")
        logger.info(f"输出文件: {output_path}")
        logger.info(f"总处理帧数: {frame_count}")
        logger.info(f"总耗时: {total_elapsed:.2f} 秒")
        logger.info(f"平均处理速度: {final_fps:.2f} FPS")
        if skip_frames > 0:
            actual_inferences = (frame_count + skip_frames) // (skip_frames + 1)
            logger.info(
                f"实际推理次数: {actual_inferences} (节省了 {(1 - actual_inferences / frame_count) * 100:.1f}% 的推理计算)")
        logger.info("=" * 30)

    except Exception as e:
        logger.error(f"发生未知错误: {e}", exc_info=True)
        # 确保资源释放
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        if 'out' in locals() and out.isOpened():
            out.release()