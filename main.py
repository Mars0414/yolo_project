import json
import cv2
import numpy as np
from ultralytics import YOLO
from detect import detect

VIDEO_PATH = 'test_video.mp4'
ZONE_FILE = 'restricted_zones.json'
OUTPUT_PATH = 'result_video.mp4'
MODEL_PATH = 'yolov8m.pt'
FRAME_STEP = 3
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

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0: fps = 30
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print(f"Обработка запущена. Всего кадров: {total_frames}")
print(f"Видео сохраняется в: {OUTPUT_PATH}")
print("Нажми 'q', если хочешь остановить раньше времени.")

alarm_frames_duration = int(fps * ALARM_DELAY_SEC)
alarm_timer = 0
count = 0

last_boxes = []
person_in_zone_last = False

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Видео закончилось (все кадры обработаны).")
            break

        count += 1

        if count % FRAME_STEP == 0:
            results = model.predict(source=frame, classes=[0], conf=CONFIDENCE, verbose=False)

            last_boxes = []
            person_in_zone_now = False

            for result in results:
                for box in result.boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, coords)
                    person_coords = [x1, y1, x2, y2]

                    is_inside = detect(person_coords, ZONE_POLYGON)
                    last_boxes.append((x1, y1, x2, y2, is_inside))

                    if is_inside:
                        person_in_zone_now = True

            person_in_zone_last = person_in_zone_now

            if person_in_zone_now:
                alarm_timer = alarm_frames_duration
            else:
                if alarm_timer > 0:
                    alarm_timer -= 1

        else:
            if alarm_timer > 0:
                alarm_timer -= 1

        for box_data in last_boxes:
            x1, y1, x2, y2, is_inside = box_data
            if is_inside:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(frame, "IN ZONE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if alarm_timer > 0:
            cv2.putText(frame, "ALARM!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 5)
            cv2.polylines(frame, [ZONE_POLYGON], isClosed=True, color=(0, 0, 255), thickness=3)
        else:
            cv2.polylines(frame, [ZONE_POLYGON], isClosed=True, color=(0, 255, 0), thickness=2)

        out.write(frame)
        cv2.imshow("Security Camera", frame)

        if count % 30 == 0:
            print(f"Обработано кадров: {count} / {total_frames}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Принудительная остановка пользователем.")
            break

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Готово! Файл {OUTPUT_PATH} успешно сохранен.")