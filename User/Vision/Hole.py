import cv2 as cv
import numpy as np

"""
1.读取图片
2.ROI提取
3.图像切割
4.图像处理
5.特征提取
6.特征匹配
7.定位
"""


def get_ROI(contours):
    ROI = []
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
        ROI.append([x, y, w, h])
    return ROI


COLOR_THRESHOLD = {
    "red": {
        "Lower": np.array([0, 28, 46]),
        "Upper": np.array([10, 255, 255]),
    },
    "red2": {
        "Lower": np.array([156, 43, 46]),
        "Upper": np.array([180, 255, 255]),
    },
    "blue": {
        "Lower": np.array([102, 96, 128]),
        "Upper": np.array([113, 255, 255]),
    },
    "yellow": {
        "Lower": np.array([24, 72, 60]),
        "Upper": np.array([35, 255, 255]),
    },
}

KERNEL = np.array(
    [
        [0, 1, 1, 0],
        [1, 1, 1, 1],
        [0, 1, 1, 0],
    ],
    np.uint8,
)

aim = "red"

video = cv.VideoCapture(1)
video.set(3, 320)
video.set(4, 240)

if not video.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = video.read()
    if not ret:
        break

    # 图像处理

    img = frame.copy()
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    inRange_hsv_r = cv.inRange(
        hsv,
        COLOR_THRESHOLD["red"]["Lower"],
        COLOR_THRESHOLD["red"]["Upper"],
    )
    inRange_hsv_r_2 = cv.inRange(
        hsv,
        COLOR_THRESHOLD["red2"]["Lower"],
        COLOR_THRESHOLD["red2"]["Upper"],
    )

    inRange_hsv_r = cv.bitwise_or(inRange_hsv_r, inRange_hsv_r_2)
    cv.imshow("red", inRange_hsv_r)
    cv.medianBlur(inRange_hsv_r, 5, inRange_hsv_r)
    cv.imshow("red_blur", inRange_hsv_r)

    inRange_aim = cv.morphologyEx(inRange_hsv_r, cv.MORPH_CLOSE, KERNEL)
    cv.imshow("aim", inRange_aim)

    contours, _ = cv.findContours(
        inRange_aim.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    ROI = []
    ROI_contours = get_ROI(contours)
    for rect in ROI_contours:
        x, y, w, h = rect
        take_h = int(h * 0.1)
        take_w = int(w * 0.1)
        retval = inRange_aim[
            y + take_h : y + h - take_h, x + take_w : x + w - take_w
        ].copy()
        retval = cv.bitwise_not(retval)
        retval = cv.morphologyEx(retval, cv.MORPH_OPEN, KERNEL)
        contours, _ = cv.findContours(
            retval.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        ROI_contours = get_ROI(contours)
        for rect in ROI_contours:
            x_, y_, w_, h_ = rect
            ROI.append([x + take_w + x_, y + take_h + y_, w_, h_])

    img_setment = inRange_aim.copy()
    for rect in ROI:
        x, y, w, h = rect
        cv.rectangle(
            inRange_aim,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

        height, width = inRange_aim.shape

        for i in range(height):
            for j in range(width):
                if y < i < y + h and x < j < x + w:
                    pass
                else:
                    img_setment[i][j] = 255

    img_setment = cv.bitwise_not(img_setment)
    img_setment = cv.erode(img_setment, KERNEL, iterations=1)
    masked_image = cv.bitwise_and(img, img, mask=img_setment)

    gray = cv.cvtColor(masked_image, cv.COLOR_BGR2GRAY)

    # 二值化处理
    _, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)

    # 反转二值图像
    binary = cv.bitwise_not(binary)
    cv.imshow("binary", binary)
    img_setment = cv.erode(img_setment, KERNEL, iterations=5)
    binary = cv.bitwise_and(binary, img_setment)

    cv.imshow("binary_only", binary)
    # 查找轮廓
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 找到面积最大的轮廓（假设它是小孔）
    contour = max(contours, key=cv.contourArea)

    # 计算轮廓的质心以找到中心
    M = cv.moments(contour)
    cx, cy = None, None
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

    # 在原始图像上绘制轮廓和中心点
    result = masked_image.copy()
    cv.drawContours(result, [contour], -1, (0, 255, 0), 2)
    cv.circle(result, (cx, cy), 1, (0, 0, 255), -1)

    # 保存并显示结果图像
    cv.imshow("result", result)

    print(f"pinhole: ({cx}, {cy})")

    cv.imshow("masked_image", masked_image)
    cv.imshow("setment", img_setment)
    cv.imshow("ROI", inRange_aim)
    cv.imshow("Image", img)
    cv.waitKey(1)
