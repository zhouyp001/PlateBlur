# PlateBlur — 车牌马赛克桌面应用

基于 PySide6 + YOLO + OpenCV 构建的 **桌面 GUI 应用**，对视频中的车牌自动检测并添加马赛克（高斯模糊），保护隐私。

## 功能

- **桌面 GUI** — 图形化界面，选择视频、一键处理、实时预览，无需命令行操作
- **自动检测** — YOLO 模型逐帧定位车牌区域
- **高斯模糊** — 对检测到的车牌区域应用马赛克效果（99x99 核）
- **跳帧加速** — 可配置跳帧间隔，每隔 N 帧推理一次，大幅提升处理速度
- **CPU / GPU 自适应** — 检测到 CUDA 自动使用 PyTorch 加速；无 GPU 回退 ONNX CPU 推理
- **实时统计** — 处理进度、速度、剩余时间、系统资源（CPU/RAM/GPU）实时显示
- **结果预览** — 处理前后视频并排对比播放

## 安装

```bash
git clone https://github.com/zhouyp001/PlateBlur.git
cd PlateBlur
pip install -r requirements.txt
```

## 依赖

| 包 | 说明 |
|---|---|
| PySide6 >= 6.0 | Qt 桌面 GUI 框架 |
| ultralytics >= 8.0 | YOLO 目标检测 |
| torch >= 1.8 | PyTorch 推理后端（GPU 使用） |
| opencv-python >= 4.0 | 视频编解码与图像处理 |
| pyyaml >= 5.0 | 配置文件解析 |
| psutil >= 5.0 | 系统资源监控 |

## 使用方式

```bash
python main.py
```

操作流程：

1. 点击「选择视频」打开待处理的视频文件
2. 确认 `config/config.yml` 中的参数（置信度、跳帧等）
3. 点击「开始处理」，实时查看处理进度和统计信息
4. 处理完成后在右侧预览结果，点击「另存为」导出

## 配置

编辑 `config/config.yml` 调整处理参数，修改后重启应用生效：

```yaml
conf: 0.15                             # 检测置信度阈值 (0.0 - 1.0)
resize: 1280                           # 推理图像尺寸
skip: 1                                # 跳帧间隔 (0 = 每帧推理)
model_pt: weights/best.pt              # PyTorch 模型 (CUDA 使用)
model_onnx: weights/best_imgsz_1280.onnx  # ONNX 模型 (CPU 使用)
```

## 项目结构

```
.
├── main.py                  # GUI 应用入口
├── gui/
│   └── main_window.py       # 主窗口、视频播放器、统计面板、资源监控
├── utils/                   # 工具模块
│   ├── processor.py         # 视频处理引擎（YOLO推理 + 马赛克 + QThread）
│   ├── logging_config.py    # 日志配置
│   └── export_onnx.py       # ONNX 模型导出脚本
├── config/
│   └── config.yml           # 用户配置文件
├── weights/                 # 模型文件
│   ├── best.pt              # YOLO 模型 (~6MB)
│   └── best_imgsz_1280.onnx # ONNX 导出模型 (~13MB)
├── tests/                   # 测试文件
│   ├── test_video_player.py
│   └── test_video_player_qt.py
├── nuitka_bat/              # Nuitka 打包脚本
│   ├── build_nuitka.bat
│   └── build_nuitka_smart.py
├── old/                     # 旧版 CLI 脚本（仅供参考）
│   ├── do_video_1.py
│   ├── do_video_2.py
│   └── do_video4onnx.py
├── requirements.txt         # Python 依赖
└── openh264-1.8.0-win64.dll # H.264 编码器
```

## 打包为独立可执行文件

```bash
# 智能打包（自动检测硬件、自适应并行编译参数）
python nuitka_bat/build_nuitka_smart.py

# 或使用批处理脚本
nuitka_bat/build_nuitka.bat
```

产物在 `dist/main.dist/main.exe`，无需安装 Python 即可运行。

## 引擎原理

YOLO 模型检测视频每帧中的车牌区域 → 对检测到的每个边界框区域应用 OpenCV `GaussianBlur`（核 99x99）→ 将模糊化后的帧编码为 MP4（H.264）输出。

跳帧模式下，非推理帧复用最近一次检测结果，在速度与精度之间取得平衡。

## License

MIT License. 详见 [LICENSE](LICENSE)。
