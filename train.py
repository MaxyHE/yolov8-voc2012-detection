from ultralytics import YOLO

model = YOLO('yolov8s.pt')

results = model.train(
    data='VOC2012.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=4,
)
