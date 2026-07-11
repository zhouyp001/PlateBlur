import os
import cv2
import time
import logging
import argparse
from pathlib import Path
from ultralytics import YOLO

# --- 性能优化设置 ---
os.environ["OMP_NUM_THREADS"] = "8"  # 根据你的CPU核心数调整
os.environ["MKL_NUM_THREADS"] = "8"

# ---------------- 配置日志 ----------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ---------------- 辅助函数 ----------------
def get_video_props(video_path):
    """获取视频属性，包含更详细的错误处理"""
    logger.info(f"正在读取视频文件: {video_path}")
    cap_ = cv2.VideoCapture(video_path)

    if not cap_.isOpened():
        logger.error(f"无法打开视频文件: {video_path}，请检查路径是否正确。")
        raise FileNotFoundError(f"Video file not found or cannot be opened: {video_path}")

    width_ = int(cap_.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_ = int(cap_.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_ = cap_.get(cv2.CAP_PROP_FPS)

    # 修复无效 FPS
    if fps_ <= 0 or fps_ > 1000:
        logger.warning("检测到无效 FPS，将默认设置为 30.0")
        fps_ = 30.0

    size_ = (width_, height_)
    size_str_ = f"{width_}x{height_}"
    total_frames = int(cap_.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"视频属性 -> 分辨率: {size_str_}, FPS: {fps_:.2f}, 总帧数: {total_frames}")
    return cap_, size_, size_str_, fps_, width_, height_, total_frames


def draw_mosaic(frame, results):
    """
    对检测到的目标区域应用马赛克（高斯模糊）
    Args:
        frame: 输入图像
        results: YOLO 模型的推理结果
    Returns:
        处理后的图像
    """
    # 如果没有结果或结果为空，直接返回原图
    if not results or len(results) == 0:
        return frame

    # 深拷贝帧，防止修改原图
    frame_copy = frame.copy()
    res = results[0]

    # 检查是否有检测框
    if res.boxes is None or len(res.boxes) == 0:
        return frame_copy

    boxes = res.boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])

        # --- 安全校验：防止坐标越界 ---
        h, w = frame_copy.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        # --- 核心步骤：提取区域并应用高斯模糊（马赛克效果）---
        roi = frame_copy[y1:y2, x1:x2]
        # 参数说明：(99, 99) 是模糊核大小，数值越大越模糊；30 是标准差
        blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
        frame_copy[y1:y2, x1:x2] = blurred_roi

    return frame_copy


# ---------------- 主程序 ----------------
if __name__ == '__main__':
    # --- 1. 定义命令行参数解析器 ---
    parser = argparse.ArgumentParser(description='视频车牌检测及马赛克处理工具')

    # 必填参数：输入文件
    parser.add_argument('input', type=str, help='输入视频文件的路径')

    # 可选参数：输出文件 (默认逻辑：输入名 + _result)
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出视频文件的路径 (默认: 输入文件名_result.mp4)')

    # 可选参数：模型缩放尺寸
    parser.add_argument('--resize', '-r', type=int, default=1280,
                        help='推理时的图像尺寸 (默认: 1280)')

    # 可选参数：置信度
    parser.add_argument('--conf', type=float, default=0.15,
                        help='检测置信度阈值 (默认: 0.15)')

    # 可选参数：跳帧
    parser.add_argument('--skip', '-s', type=int, default=1,
                        help='跳帧数量 (0为不跳帧, 1为每2帧处理1次, 默认: 1)')

    # 可选参数：模型路径
    parser.add_argument('--model', type=str, default='best.pt',
                        help='模型文件路径 (默认: best.pt)')

    # 解析参数
    args = parser.parse_args()

    # --- 2. 处理默认输出文件名逻辑 ---
    input_path = Path(args.input)
    if args.output is None:
        # 如果未指定输出，使用输入文件名 + _result
        output_path = input_path.parent / f"{input_path.stem}_result{input_path.suffix}"
    else:
        output_path = Path(args.output)

    # --- 3. 打印运行配置 ---
    logger.info("🚀 启动视频处理任务")
    logger.info(f"📄 输入: {args.input}")
    logger.info(f"💾 输出: {output_path}")
    logger.info(f"⚙️  配置: Resize={args.resize}, Conf={args.conf}, Skip={args.skip}")

    # --- 4. 初始化变量 ---
    cap = None
    out = None
    last_results = None  # 缓存上一次的检测结果，用于跳帧绘制

    try:
        # 1. 加载模型
        logger.info(f"正在加载 YOLO 模型: {args.model}")
        start_load_time = time.time()
        model = YOLO(args.model)
        load_time = time.time() - start_load_time
        logger.info(f"模型加载完成，耗时: {load_time:.2f} 秒")

        # 2. 获取视频属性
        cap, size, size_str, fps, width, height, total_frames = get_video_props(args.input)

        # 3. 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        logger.info(f"正在初始化视频写入器: {output_path} ({size_str}, {fps:.2f} FPS)")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not out.isOpened():
            logger.error("无法创建视频写入器，请检查磁盘空间或权限。")
            raise RuntimeError("Cannot open VideoWriter")

        # 4. 开始处理循环
        logger.info("开始处理视频帧...")
        if args.skip > 0:
            logger.info(f"【模式】跳帧处理：每 {args.skip + 1} 帧推理 1 次")
        else:
            logger.info("【模式】全帧处理")

        start_time = time.time()
        frame_count = 0
        inference_count = 0  # 统计实际推理次数

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("视频读取结束或流中断。")
                break

            frame_count += 1
            should_infer = False

            # --- 核心跳帧逻辑 ---
            if args.skip == 0:
                should_infer = True
            else:
                if (frame_count - 1) % (args.skip + 1) == 0:
                    should_infer = True

            if should_infer:
                # 执行推理
                results = model.predict(
                    source=frame,
                    device="cpu",  # 如需GPU，请根据环境修改
                    verbose=False,
                    conf=args.conf,
                    imgsz=args.resize,
                    save=False,
                    show=False,
                    augment=False,
                    half=False
                )
                last_results = results[0]  # 更新缓存结果
                inference_count += 1

            # --- 绘制与写入 ---
            processed_frame = draw_mosaic(frame, [last_results] if last_results else None)
            out.write(processed_frame)

            # --- 进度日志 ---
            if frame_count % max(1, total_frames // 20) == 0:
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                progress = (frame_count / total_frames) * 100
                logger.info(f"进度: {progress:.1f}% | 帧: {frame_count}/{total_frames} | "
                            f"推理: {inference_count}次 | 速度: {current_fps:.2f} FPS")

        # 5. 收尾与总结
        total_elapsed = time.time() - start_time
        final_fps = frame_count / total_elapsed if total_elapsed > 0 else 0

        logger.info("=" * 50)
        logger.info("🎉 处理完成!")
        logger.info(f"输出文件: {output_path}")
        logger.info(f"总帧数: {frame_count} | 推理次数: {inference_count}")
        logger.info(f"总耗时: {total_elapsed:.2f}秒 | 平均速度: {final_fps:.2f} FPS")
        if args.skip > 0:
            saved_ratio = ((frame_count - inference_count) / frame_count) * 100
            logger.info(
                f"性能提升: 跳过了 {(frame_count - inference_count)} 帧推理，节省了约 {saved_ratio:.1f}% 的计算量")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"程序发生致命错误: {e}", exc_info=True)
    finally:
        # 确保资源释放
        if 'cap' in locals() and cap and cap.isOpened():
            cap.release()
        if 'out' in locals() and out and out.isOpened():
            out.release()
        logger.info("视频资源已释放。")