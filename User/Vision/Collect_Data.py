import cv2 as cv
import numpy as np
from pprint import pprint
import time

# import pandas as pd
from Threshold_Editor import editor

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
# 定义文件名
index = "red&blue"
data_log = {
    "frame": [],
    "color": [],
    "x": [],
    "y": [],
    "approx": [],
    "area": [],
    "area_ratio": [],
}
# 设置
flag_video = 0  # 0为图片，1为视频
flag_open: bool = False  # 能否正确打开摄像头
flag_collect = 0  # 是否开始采集数据
collect_index = "collect_"  # 采集数据的索引
collect_num = 0  # 采集数据的数量
flag_collected = 0  # 是否采集到数据

# 主函数
if __name__ == "__main__":

    # 打开摄像头
    if flag_video:
        cap = cv.VideoCapture(0)
        # cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
        # cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv.CAP_PROP_FPS, 20)
        cap.set(3, 320)  # 设置分辨率480p
        cap.set(4, 240)
        flag_open = cap.isOpened()  # 能否正确打开摄像头
        if not flag_open:
            print("打不开摄像头")

    while flag_open or not flag_video:
        # 读取图像，摄像头或者图片
        if flag_video:
            ret, img = cap.read()
        else:
            img = cv.imread(r"E:/project/User/Things/red&blue.jpg")
            # img = cv.resize(img, (128, 96))
        cv.imshow("zero", img)
        now_time = time.time()

        # 高斯模糊 和 HSV空间的转换
        # gs_img = cv.GaussianBlur(img, (3, 3), 0)
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # 颜色范围的掩膜,获取所需颜色的掩膜
        inRange_hsv_r = cv.inRange(
            hsv, color_dict["red"]["Lower"], color_dict["red"]["Upper"]
        )
        inRange_hsv_b = cv.inRange(
            hsv, color_dict["blue"]["Lower"], color_dict["blue"]["Upper"]
        )
        inRange_hsv_y = cv.inRange(
            hsv, color_dict["yellow"]["Lower"], color_dict["yellow"]["Upper"]
        )
        inRange_hsv_r_2 = cv.inRange(
            hsv, color_dict["red2"]["Lower"], color_dict["red2"]["Upper"]
        )
        inRange_hsv_r = cv.bitwise_or(inRange_hsv_r, inRange_hsv_r_2)

        # cv.imshow("inRange_hsv_r", inRange_hsv_r)
        # cv.imshow("inRange_hsv_b", inRange_hsv_b)
        # cv.imshow("inRange_hsv_y", inRange_hsv_y)

        # 合成目标二值化图像
        inRange_aim = cv.bitwise_or(inRange_hsv_r, inRange_hsv_b)
        inRange_aim = cv.bitwise_or(inRange_aim, inRange_hsv_y)

        kernel_circle = np.array(
            [
                [1, 1, 1, 1],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [1, 1, 1, 1],
            ],
            np.uint8,
        )

        inRange_aim = cv.medianBlur(inRange_aim, 5)
        inRange_aim = cv.morphologyEx(inRange_aim.copy(), cv.MORPH_CLOSE, kernel_circle)
        # cv.imshow("aim", inRange_aim)

        # 查找轮廓
        contours, _ = cv.findContours(
            inRange_aim.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        del inRange_aim

        # 遍历轮廓
        for contour in contours:
            # 获取轮廓位置
            # rect = cv.minAreaRect(contour)
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
            print(x, y, w, h)
            # 圈定识别范围ROI
            # cv.rectangle(show_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # cv.putText(
            #     show_img,
            #     str(round(area_ratio, 2)),
            #     (x + 80, y - 2),
            #     cv.FONT_HERSHEY_COMPLEX,
            #     0.5,
            #     (0, 0, 255),
            #     1,
            # )

            # 颜色识别
            color = "unknown"
            step = 10  # 颜色识别采样的步长
            color_judge = {"red": 0, "blue": 0, "yellow": 0}

            # 代码待定，太繁杂
            for i in range(x, x + w, step):
                for j in range(y, y + h, step * 2):
                    for color_name, hsv_range in color_dict.items():
                        lower_bound = hsv_range["Lower"]
                        upper_bound = hsv_range["Upper"]
                        if np.all(lower_bound <= hsv[j, i]) and np.all(
                            hsv[j, i] <= upper_bound
                        ):
                            if color_name == "red2":
                                color_name = "red"
                            color_judge[color_name] += 1
                            # print(color_name)
                            break
                    else:
                        continue
            color = max(color_judge, key=color_judge.get)
            # cv.putText(
            #     show_img,
            #     color,
            #     (x, y - 2),
            #     cv.FONT_HERSHEY_COMPLEX,
            #     0.5,
            #     (0, 0, 255),
            #     1,
            # )

            # cv.circle(show_img, (int(rect[0][0]), int(rect[0][1])), 1, (0, 0, 0), -1)
            # 识别形状
            approx = cv.approxPolyDP(contour, 0.025 * cv.arcLength(contour, True), True)
            # cv.drawContours(show_img, [approx], 0, (255, 0, 0), 1)

            # cv.putText(
            #     show_img,
            #     str(len(approx)),
            #     (x + 60, y - 2),
            #     cv.FONT_HERSHEY_COMPLEX,
            #     0.5,
            #     (255, 0, 0),
            #     1,
            # )
            # cv.imshow("show", show_img)
            print(time.time() - now_time)
            print(color, str(len(approx)))
            if flag_collect == 1:
                data_log["frame"].append(collect_num)
                data_log["color"].append(color)
                data_log["x"].append(x)
                data_log["y"].append(y)
                data_log["approx"].append(len(approx))
                data_log["area"].append(w * h)
                data_log["area_ratio"].append(area_ratio)
                flag_collected = 1
        collect_num += 1
        k = cv.waitKey(1) & 0xFF
        if k == ord("s"):  # 按下s键，进入下面的保存图片操作
            cv.imwrite(r"../Things/" + index + ".jpg", img)
            editor()
            print("save" + str(index) + ".jpg successfully!")
            # elif k == ord("q"):  # 按下q键，程序退出
            # pd.DataFrame(data_log).to_csv(r"E:/ultimately/Vision/Things/Log/" + "data_log.csv", index=False)
            break
        elif flag_collected == 1:
            cv.imwrite(
                r"../Things/Log/" + collect_index + str(collect_num) + "_imshow.jpg",
                show_img,
            )
            cv.imwrite(
                r"../Things/Log"
                + collect_index
                + str(collect_num)
                + "_inRange_hsv_r.jpg",
                inRange_hsv_r,
            )
            cv.imwrite(
                r"../Things/Log/"
                + collect_index
                + str(collect_num)
                + "_inRange_hsv_b.jpg",
                inRange_hsv_b,
            )
            cv.imwrite(
                r"../Things/Log/"
                + collect_index
                + str(collect_num)
                + "_inRange_hsv_y.jpg",
                inRange_hsv_y,
            )
            flag_collected = 0
        elif not flag_video:
            while True:
                k = cv.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
            break

    cv.destroyAllWindows()
