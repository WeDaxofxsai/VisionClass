import cv2 as cv
import numpy as np
import time


def callBackInRange(x):
    global L_H, L_S, L_V, H_H, H_S, H_V
    L_H = cv.getTrackbarPos("L_H", "inRange")
    L_S = cv.getTrackbarPos("L_S", "inRange")
    L_V = cv.getTrackbarPos("L_V", "inRange")

    H_H = cv.getTrackbarPos("H_H", "inRange")
    H_S = cv.getTrackbarPos("H_S", "inRange")
    H_V = cv.getTrackbarPos("H_V", "inRange")


def editor():
    global opening_hsv, aim, L_H, L_S, L_V, H_H, H_S, H_V
    L_H = 0
    L_S = 0
    L_V = 0

    H_H = 0
    H_S = 0
    H_V = 0
    cv.namedWindow("inRange")

    cv.createTrackbar("L_H", "inRange", 0, 179, callBackInRange)
    cv.createTrackbar("L_S", "inRange", 0, 256, callBackInRange)
    cv.createTrackbar("L_V", "inRange", 0, 256, callBackInRange)

    cv.createTrackbar("H_H", "inRange", 0, 179, callBackInRange)
    cv.createTrackbar("H_S", "inRange", 0, 256, callBackInRange)
    cv.createTrackbar("H_V", "inRange", 0, 256, callBackInRange)

    editor_img = cv.imread("E:/project/User/Things/red&blue.jpg")
    # gsImg = cv.GaussianBlur(editor_img, (3, 3), 0)
    hsv = cv.cvtColor(editor_img, cv.COLOR_BGR2HSV)
    opening_hsv = hsv.copy()

    aim = cv.inRange(opening_hsv, np.array([0, 0, 0]), np.array([0, 0, 0]))

    # 90 58 171 104
    while True:
        # ret, editor_img = cap.read()

        cv.imshow("editor_img", editor_img)
        # gsImg = cv.GaussianBlur(editor_img, (3, 3), 0)
        hsv = cv.cvtColor(editor_img, cv.COLOR_BGR2HSV)
        opening_hsv = hsv.copy()
        inRange_aim = cv.inRange(
            opening_hsv, np.array([L_H, L_S, L_V]), np.array([H_H, H_S, H_V])
        )

        cv.imshow("inRange", inRange_aim)
        cv.imshow("show", inRange_aim)

        inRange_aim = cv.medianBlur(inRange_aim, 5)
        inRange_aim = cv.morphologyEx(
            inRange_aim.copy(),
            cv.MORPH_CLOSE,
            np.array(
                [
                    [0, 1, 1, 0],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                    [0, 1, 1, 0],
                ],
                np.uint8,
            ),
        )
        cv.imshow("aim", inRange_aim)

        # 查找轮廓
        contours = cv.findContours(
            inRange_aim.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )[0]
        del inRange_aim

        show_img = editor_img.copy()
        for contour in contours:
            # 获取轮廓位置
            rect = cv.minAreaRect(contour)
            x, y, w, h = cv.boundingRect(contour)
            out_area = w * h
            in_area = cv.contourArea(contour)
            area_ratio = in_area / out_area

            # 限制条件
            if (
                w * h < 1000
                or w * h > 45000
                or w < 20
                or h < 20
                or (abs(w - h) / max(w, h)) > 0.6
                or area_ratio < 0.60
            ):
                continue

            # 圈定识别范围ROI
            cv.rectangle(show_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cv.rectangle(show_img, (90, 58), (90 + 171, 58 + 104), (0, 0, 255), 3)

            cv.rectangle(
                show_img,
                (90 - int(171 / 10), 58 - int(104 / 10)),
                (90 + 171 + int(171 / 10), 58 + 104 + int(104 / 10)),
                (255, 255, 0),
                1,
            )
            cv.putText(
                show_img,
                str(round(in_area / (171 * 104), 4)),
                (x + 80, y - 2),
                cv.FONT_HERSHEY_COMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

            cv.circle(show_img, (int(rect[0][0]), int(rect[0][1])), 1, (0, 0, 0), -1)
            # 识别形状
            approx = cv.approxPolyDP(contour, 0.025 * cv.arcLength(contour, True), True)
            cv.drawContours(show_img, [approx], 0, (255, 0, 0), 1)

            cv.putText(
                show_img,
                str(len(approx)),
                (x + 60, y - 2),
                cv.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

            cv.putText(  # 90 58 171 104
                show_img,
                str(round(in_area / ((171 * 1.2) * (104 * 1.2)), 4)),
                (x + 80, y - 12),
                cv.FONT_HERSHEY_COMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

            cv.imshow("show_img", show_img)

        if cv.waitKey(1) == ord("q"):
            print(L_H, L_S, L_V, H_H, H_S, H_V)

    cv.destroyAllWindows()


if __name__ == "__main__":
    editor()
