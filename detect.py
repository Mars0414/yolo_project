import cv2


def detect(person_coords, zone_polygon):
    x1, y1, x2, y2 = map(int, person_coords)

    height = y2 - y1

    y_levels = [
        int(y2),
        int(y2 - (height * 0.25)),
        int(y2 - (height * 0.50))
    ]

    x_points = [x1, int((x1 + x2) / 2), x2]

    for y in y_levels:
        for x in x_points:
            if cv2.pointPolygonTest(zone_polygon, (x, y), False) >= 0:
                return True

    return False