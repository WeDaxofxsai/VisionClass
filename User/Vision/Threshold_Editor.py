import cv2 as cv
import numpy as np


def callBackInRange(x):
    global opening_hsv, aim, L_H, L_S, L_V, H_H, H_S, H_V
    L_H = cv.getTrackbarPos("L_H", "inRange")
    L_S = cv.getTrackbarPos("L_S", "inRange")
    L_V = cv.getTrackbarPos("L_V", "inRange")

    H_H = cv.getTrackbarPos("H_H", "inRange")
    H_S = cv.getTrackbarPos("H_S", "inRange")
    H_V = cv.getTrackbarPos("H_V", "inRange")

    # aim = cv.inRange(opening_hsv, np.array([L_H, L_S, L_V]), np.array([H_H, H_S, H_V]))


def editor():
    global opening_hsv, aim, L_H, L_S, L_V, H_H, H_S, H_V
    L_H = 0
    L_S = 0
    L_V = 0

    H_H = 0
    H_S = 0
    H_V = 0

    cap = cv.VideoCapture(0)
    editor_img = cv.imread("../Things/red&blue.jpg")
    # editor_img = cv.resize(editor_img, (640, 480))

    img1 = editor_img.copy()

    gsImg = cv.GaussianBlur(editor_img, (3, 3), 0)
    hsv = cv.cvtColor(gsImg, cv.COLOR_BGR2HSV)
    opening_hsv = hsv.copy()
    cv.imshow("hsv", hsv)
    # cv.imshow('opening_hsv', opening_hsv)
    cv.namedWindow("inRange")

    cv.createTrackbar("L_H", "inRange", 0, 179, callBackInRange)
    cv.createTrackbar("L_S", "inRange", 0, 256, callBackInRange)
    cv.createTrackbar("L_V", "inRange", 0, 256, callBackInRange)

    cv.createTrackbar("H_H", "inRange", 0, 179, callBackInRange)
    cv.createTrackbar("H_S", "inRange", 0, 256, callBackInRange)
    cv.createTrackbar("H_V", "inRange", 0, 256, callBackInRange)

    aim = cv.inRange(opening_hsv, np.array([0, 0, 0]), np.array([0, 0, 0]))

    while True:
        _, img1 = cap.read()
        gsImg = cv.GaussianBlur(img1, (3, 3), 0)
        hsv = cv.cvtColor(gsImg, cv.COLOR_BGR2HSV)
        opening_hsv = hsv.copy()
        aim = cv.inRange(
            opening_hsv, np.array([L_H, L_S, L_V]), np.array([H_H, H_S, H_V])
        )
        cv.imshow("test", img1)
        cv.imshow("inRange", aim)
        if cv.waitKey(1) == ord("q"):
            break
    cv.destroyAllWindows()


if __name__ == "__main__":
    editor()
