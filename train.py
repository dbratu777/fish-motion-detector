from ultralytics import YOLO

model = YOLO("custom.pt")

if __name__ == '__main__':
    # model.train(data="data.yaml", imgsz=1024, batch=8, 
    #     epochs=25, workers=1, device=0)
    model.val()
