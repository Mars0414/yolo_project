from ultralytics import YOLO

model = YOLO('yolov8x.pt')

results = model.predict(source='test_video.mp4', conf=0.25, iou=0.4, save=True, classes=[0])

for r in results:
    print(r.verbose())
