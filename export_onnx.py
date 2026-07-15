from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('best.pt')
    model.export(format='onnx', imgsz=1280, simplify=True)