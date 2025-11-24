import cv2
import numpy as np
import json

img = cv2.imread(filename='zone.png')

if img is None:
    print("No image")
    exit()

hsv = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

mask = mask1 + mask2

contours, _ = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

if contours:
    zone_contour = max(contours, key=cv2.contourArea)

    cv2.drawContours(image=img, contours=[zone_contour], contourIdx=-1, color=(0, 255, 0), thickness=3)

    points_list = zone_contour.reshape(-1, 2).tolist()

    date_to_save = {
        'zone1': points_list
    }

    output_file = 'zone_coords.json'
    with open(output_file, 'w') as f:
        json.dump(date_to_save, f, indent=4)

    print(f'zone_coords.json saved to {output_file}')
    print(f'number of points: {len(points_list)}')

else:
    print('no zones')

cv2.imshow('Mask', mat=mask)
cv2.imshow('Result', mat=img)
cv2.waitKey(0)
cv2.destroyAllWindows()