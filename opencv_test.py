import cv2
import os

video_path = 'test_video.mp4'
output_folder = 'frames_output'
step = 100

if not os.path.exists(output_folder):
    os.makedirs(name=output_folder)

    cap = cv2.VideoCapture(video_path)
    count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break


        if count % step == 0:
            filename = os.path.join(output_folder, 'frame%d.jpg' % count)
            cv2.imwrite(filename=filename, img=frame)
            saved_count += 1


        count += 1

    cap.release()
    print(f"Обработка кадров: {count}")
    print(f"Сохранено изображений: {saved_count}")