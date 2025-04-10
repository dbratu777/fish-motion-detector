from ultralytics import YOLO

try:
    model = YOLO("custom.pt")
except Exception:
    model = YOLO("yolo11n.pt")

if __name__ == '__main__':
    model.train(data="data.yaml", imgsz=1024, batch=8,
                epochs=100, workers=1, device=0)
    model.val()
