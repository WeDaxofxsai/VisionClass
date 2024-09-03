import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


class aim_object:
    def __init__(self, _color, _x, _y, _w, _h):
        self.color = _color
        self.x = _x
        self.y = _y
        self.w = _w
        self.h = _h
        self.area = 0
        self.color_ratio = 0
        self.area_ratio = 0


# 定义颜色范围的字典
color_dict = {
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
# 读取图像
original_img = cv.imread("E:/project/User/Things/red&blue.jpg")


# 转换图像空间
img = original_img.copy()
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

h, s, v = cv.split(hsv)
v_eq = cv.equalizeHist(v)

# 3. 将均衡化后的V通道与H和S通道合并
hsv = cv.merge([h, s, v_eq])

# 4. 将均衡化后的图像转换回BGR颜色空间
# image_eq = cv2.cvtColor(hsv_image_eq, cv2.COLOR_HSV2BGR)


# 颜色阈值的获取
inRange_aim = cv.inRange(
    hsv.copy(), np.array([L_H, L_S, L_V]), np.array([H_H, H_S, H_V])
)
# 双边滤波
inRange_aim = cv.medianBlur(inRange_aim, 5)
# 形态学操作
inRange_aim = cv.morphologyEx(
    inRange_aim,
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

# 获取目标位置
contours, _ = cv.findContours(
    inRange_aim.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
)

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

    # 指标参数计算
    ## 外围圈定点的获取

    # 判断颜色
    color_judge = {
        "red": 0,
        "blue": 0,
        "yellow": 0,
    }

    step_color = 10  # 颜色识别采样的步长
    for i in range(x, x + w, step_color):
        for j in range(y, y + h, step_color * 2):
            for color_name, hsv_range in self.COLOR_THRESHOLD.items():
                lower_bound = hsv_range["Lower"]
                upper_bound = hsv_range["Upper"]
                if np.all(lower_bound <= opening_hsv[j, i]) and np.all(
                    opening_hsv[j, i] <= upper_bound
                ):
                    if color_name == "red2":
                        color_name = "red"
                    color_judge[color_name] += 1
                    break
            else:
                continue
    a_color = max(color_judge, key=color_judge.get)

    temp_threshold = color_dict[a_color]["Lower"]
    count_iteration = 0
    for i in range(count_iteration):
        pass
