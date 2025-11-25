import json
import cv2
import numpy as np
from ultralytics import YOLO
from detect import detect

VIDEO_PATH = 'test_video.mp4'
ZONE_FILE = 'restricted_zones.json'
MODEL_PATH = 'yolov8m.pt'
FRAME_STEP = 2
ALARM_DELAY_SEC = 3
CONFIDENCE = 0.40

print(f"Загружаем зону из {ZONE_FILE}...")
try:
    with open(ZONE_FILE, 'r') as f:
        data = json.load(f)
        zone_points = list(data.values())[0]
        ZONE_POLYGON = np.array(zone_points, np.int32)
except Exception as e:
    print(f"ОШИБКА: {e}")
    exit()


model = YOLO(MODEL_PATH)


cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ОШИБКА: Не удалось открыть видео!")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30
alarm_frames_duration = int(fps * ALARM_DELAY_SEC)
alarm_timer = 0

count = 0
print("Запуск режима повышенной чувствительности...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    count += 1
    if count % FRAME_STEP != 0:
        if alarm_timer > 0:
            alarm_timer -= 1
        continue


    results = model.predict(source=frame, classes=[0], conf=CONFIDENCE, verbose=False)

    person_in_zone_now = False

    for result in results:
        boxes = result.boxes
        for box in boxes:
            coords = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, coords)
            person_coords = [x1, y1, x2, y2]


            if detect(person_coords, ZONE_POLYGON):
                person_in_zone_now = True


                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "IN ZONE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if person_in_zone_now:
        alarm_timer = alarm_frames_duration
    else:
        if alarm_timer > 0:
            alarm_timer -= FRAME_STEP

    if alarm_timer > 0:
        cv2.putText(frame, "ALARM!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)
        cv2.polylines(frame, [ZONE_POLYGON], isClosed=True, color=(0, 0, 255), thickness=3)
    else:
        cv2.polylines(frame, [ZONE_POLYGON], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow("Security Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()