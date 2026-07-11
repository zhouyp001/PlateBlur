# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A video processing tool that detects license plates using YOLO object detection and applies Gaussian blur (mosaic) to hide them. Runs on CPU with performance optimizations (skip-frame, ONNX export).

## Main Scripts

- **`do_video_2.py`** — Production script with full argparse CLI. Use this as the primary entry point.
- **`do_video4onnx.py`** — Variant optimized for ONNX model inference on CPU (`best_imgsz_1280.onnx`).
- **`do_video_1.py`** — Early version with hardcoded paths; kept for reference only.

## Commands

```bash
# Run detection on a video
python do_video_2.py input.mp4 --output result.mp4 --resize 1280 --conf 0.15 --skip 1

# Export model to ONNX for faster CPU inference
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx', imgsz=1280, simplify=True)"

# Package as one-file executable (with model bundled)
pyinstaller -F -c -n "hide-license" --add-data "best.pt;." do_video_2.py

# Package as one-dir executable
pyinstaller --onedir -c -n "hide-license" --add-data "best.pt;." do_video_2.py
```

## Key Architecture

- **Detection pipeline**: YOLO model detects license plates → `draw_mosaic()` applies `cv2.GaussianBlur` with kernel (99, 99) to each detected bounding box.
- **Skip-frame optimization**: Default `--skip 1` means only every other frame runs YOLO inference; skipped frames reuse the last detection result.
- **Model files**: `best.pt` is the PyTorch model (~6MB), `best_imgsz_1280.onnx` is the ONNX export (~13MB) optimized for imgsz=1280.
- **Vendored ultralytics**: The `ultralytics/` directory is a local copy of the library used at import time — modifications here affect detection behavior.
- **Packaging**: PyInstaller bundles the model file and Python runtime into a standalone `.exe`. The `openh264-1.8.0-win64.dll` is the H.264 codec needed for MP4 output.

## CLI Arguments (do_video_2.py)

| Arg | Default | Description |
|-----|---------|-------------|
| `input` | (required) | Input video path |
| `--output` / `-o` | `{input}_result.mp4` | Output video path |
| `--resize` / `-r` | `1280` | Inference image size |
| `--conf` | `0.15` | Detection confidence threshold |
| `--skip` / `-s` | `1` | Frames to skip between inferences (0 = every frame) |
| `--model` | `best.pt` | Model file path |
