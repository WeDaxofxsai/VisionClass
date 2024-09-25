import cv2 as cv
import numpy as np
import time

# 84.55433655   0.         208.43049622], Upper = [144.55433655  40.60594177 255
COLOR_THRESHOLD = {
    "white": {
        "Lower": np.array([84, 0, 208]),
        "Upper": np.array([144, 40, 255]),
    },
    "red": {
        "Lower": np.array([0, 43, 46]),
        "Upper": np.array([10, 255, 255]),
    },
    "red2": {
        "Lower": np.array([156, 43, 46]),
        "Upper": np.array([180, 255, 255]),
    },
    "blue": {
        "Lower": np.array([85, 46, 221]),
        "Upper": np.array([124, 255, 255]),
    },
    "yellow": {
        "Lower": np.array([24, 72, 60]),
        "Upper": np.array([35, 255, 255]),
    },
}
kernel_circle = np.array(
    [
        [0, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 0],
    ],
    np.uint8,
)

# image = cv.imread(r"E:\project\User\Things\white_ball.jpg")
# cv.imshow("Original Image", image)

capture = cv.VideoCapture(1)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
while True:
    ret, image = capture.read()
    # cv.imshow("Video", image)

    # start_time = time.time()
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    inRange_hsv = cv.inRange(
        hsv,
        COLOR_THRESHOLD["white"]["Lower"],
        COLOR_THRESHOLD["white"]["Upper"],
    )
    # cv.imshow("White Ball", inRange_hsv)

    inRange_aim = cv.morphologyEx(inRange_hsv, cv.MORPH_CLOSE, kernel_circle)
    # cv.imshow("Aim", inRange_aim)
    inRange_aim = cv.medianBlur(inRange_hsv, 5)
    cv.imshow("Aim", inRange_aim)
    
    contours, _ = cv.findContours(
                inRange_aim, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        # 获取轮廓位置
        rect = cv.minAreaRect(contour)
        x, y, w, h = cv.boundingRect(contour)
        out_area = w * h  # 外部面积
        in_area = cv.contourArea(contour)  # 内部面积
        area_ratio = in_area / out_area  # 面积比

        # 过滤小面积
        if (
            w * h < 2000  # 面积小于2000
            or w * h > 60000  # 面积大于60000
            or w < 15
            or h < 15  # 宽度或高度小于20
            or (abs(w - h) / max(w, h)) > 0.6
            or area_ratio < 0.60
        ):  # 面积比小于0.6
            continue
        cv.circle(image, (int(rect[0][0]), int(rect[0][1])), 1, (0, 0, 0), -1)
        cv.circle(image, (int(rect[0][0]), int(rect[0][1])), int(w/2), (0, 0, 255), 2)
        approx = cv.approxPolyDP(
            contour, 0.025 * cv.arcLength(contour, True), True
        )
        cv.putText(
            image,
            str(len(approx)),
            (x + 60, y - 2),
            cv.FONT_HERSHEY_COMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )
    cv.imshow("Result", image)
    cv.waitKey(1)

