import cv2 as cv
import numpy as np

img = cv.imread("E:\\project\\User\\Things\\red&blue.jpg")
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

ret, mask = cv.threshold(gray_img, 220, 255, cv.THRESH_BINARY)

edges = cv.Canny(gray_img, 50, 150, apertureSize=3)

# 霍夫变换检测直线
lines = cv.HoughLinesP(
    edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
)

# 绘制直线
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 显示结果
cv.imshow("HoughLinesP", img)
cv.imshow("Original Image", img)
cv.imshow("Gray Image", gray_img)
cv.imshow("Mask", mask)
cv.waitKey(0)
