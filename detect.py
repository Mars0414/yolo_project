import cv2

def detect(person_coords, zone_polygon):
        x1, y1, x2, y2 = map(int, person_coords)

        feet_y = y2

        left_foot_x = x1
        right_foot_x = x2
        center_foot_x = int((x1 + x2) / 2)

        check1 = cv2.pointPolygonTest(zone_polygon, (left_foot_x, feet_y), False)
        check2 = cv2.pointPolygonTest(zone_polygon, (right_foot_x, feet_y), False)
        check3 = cv2.pointPolygonTest(zone_polygon, (center_foot_x, feet_y), False)

        if check1 >= 0 or check2 >= 0 or check3 >= 0:
            return True

        return False