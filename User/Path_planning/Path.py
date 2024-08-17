import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pprint import pprint
from pylab import *
import math

"""
地图大小为5000*5000, 车子大小为400*400,
"""

image_Color = cv.imread("User\Things\map_.original.jpg")

print("img shape:")
print("shape =", image_Color.shape)  # 打印彩色图像的（垂直像素，水平像素，通道数）
print("size =", image_Color.size)  # 打印彩色图像包含的像素个数
print("dtype =", image_Color.dtype)  # 打印彩色图像的数据类型
cv.imshow("original", image_Color)

image_Gray = cv.imread(
    "User\Things\map_.original.jpg", 0
)  # 读取与3.1.jpg（彩色图像）对应的灰度图像
print("img shape:")
print("shape =", image_Gray.shape)  # 打印灰度图像的（垂直像素，水平像素）
print("size =", image_Gray.size)  # 打印灰度图像包含的像素个数
cv.imshow("Gray", image_Gray)

ret, mask = cv.threshold(image_Gray, 180, 255, cv.THRESH_BINARY_INV)


contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
obstacle = []  # 障碍物的坐标, 长宽 格式为[(x, y), w, h]
for contour in contours:
    # 获取轮廓位置
    rect = cv.minAreaRect(contour)
    x, y, w, h = cv.boundingRect(contour)
    print(x, y, w, h)
    cv.rectangle(image_Color, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 255, 0), 1)
    cv.putText(
        image_Color,
        "[[" + str(x) + "," + str(y) + "], " + str(w) + ", " + str(h) + "]",
        (x - 5, y - 2),
        cv.FONT_HERSHEY_COMPLEX,
        0.4,
        (0, 0, 255),
        1,
    )
    obstacle.append([[x, y], w, h])

print("cross_list:")
cross_list = []
for i in range(700, 5000, 1200):
    for j in range(700, 5000, 1200):
        cross_list.append([int(i / 10), int(j / 10)])
pprint(cross_list)

print("obstacle_list:")
pprint(obstacle)
cv.imshow("result", image_Color)
cv.imshow("mask", mask)
cv.waitKey(0)
cv.destroyAllWindows()

"""Learing"""
