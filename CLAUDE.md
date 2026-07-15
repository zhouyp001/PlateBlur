# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A desktop GUI application (PySide6) that detects license plates in videos using YOLO object detection and applies Gaussian blur (mosaic) to hide them. Supports CPU/GPU auto-detection with performance optimizations (skip-frame, ONNX export).

## Main Scripts

- **`main.py`** — GUI application entry point (PySide6 desktop app).
- **`processor.py`** — Video processing engine (YOLO inference + mosaic + QThread).
- **`gui/main_window.py`** — Main window, video player, stats panel, resource monitor.
- **`old/do_video_2.py`** — Legacy CLI script with argparse (kept for reference).
- **`old/do_video4onnx.py`** — Legacy CLI ONNX variant (kept for reference).
- **`old/do_video_1.py`** — Early prototype with hardcoded paths (kept for reference).

## Commands

```bash
# Run GUI application
python main.py

# Export model to ONNX for faster CPU inference
python export_onnx.py

# Package with Nuitka (smart build with hardware detection)
python nuitka_bat/build_nuitka_smart.py
```

## Key Architecture

- **Detection pipeline**: YOLO model detects license plates → `draw_mosaic()` applies `cv2.GaussianBlur` with kernel (99, 99) to each detected bounding box.
- **Skip-frame optimization**: Default `--skip 1` means only every other frame runs YOLO inference; skipped frames reuse the last detection result.
- **Model files**: `weights/best.pt` is the PyTorch model (~6MB), `weights/best_imgsz_1280.onnx` is the ONNX export (~13MB) optimized for imgsz=1280.
- **Vendored ultralytics**: The `ultralytics/` directory is a local copy of the library used at import time — modifications here affect detection behavior.
- **Packaging**: Nuitka bundles the model file and Python runtime into a standalone `.exe`. The `openh264-1.8.0-win64.dll` is the H.264 codec needed for MP4 output.
