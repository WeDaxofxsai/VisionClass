import numpy as np
import cv2 as cv

COLOR_THRESHOLD = {
    # "green": {
    #     "Lower": np.array([0, 24, 68]),
    #     "Upper": np.array([92, 136, 140]),
    # },
    # "white": {
    #     "Lower": np.array([0, 0, 150]),
    #     "Upper": np.array([180, 40, 255]),
    # },
    "red": {
        "Lower": np.array([0, 43, 46]),
        "Upper": np.array([10, 255, 255]),
    },
    "red2": {
        "Lower": np.array([156, 43, 46]),
        "Upper": np.array([180, 255, 255]),
    },
    "blue": {
        "Lower": np.array([102, 43, 46]),
        "Upper": np.array([113, 255, 255]),
    },
    "yellow": {
        "Lower": np.array([24, 72, 60]),
        "Upper": np.array([35, 255, 255]),
    },
}  # 颜色阈值
image = cv.imread(r"E:\project\User\Things\all.jpg")
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
h, s, v = cv.split(hsv)
cv.imshow("h", h)
cv.imshow("s", s)
cv.imshow("v", v)


cv.waitKey(0)
