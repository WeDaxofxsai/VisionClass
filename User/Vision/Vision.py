import cv2 as cv
import time
import numpy as np
import threading

from pyzbar import pyzbar

# from Servo_control import qiu


class Vision:
    def __init__(self):
        self.a = 0
        self.COLOR_THRESHOLD = {
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
                "Lower": np.array([102, 96, 128]),
                "Upper": np.array([113, 255, 255]),
            },
            # "yellow": {
            #     "Lower": np.array([14, 86, 117]),
            #     "Upper": np.array([50, 255, 255]),
            # },
            "yellow": {
                "Lower": np.array([24, 72, 60]),
                "Upper": np.array([35, 255, 255]),
            },
        }  # 颜色阈值
        self.__frame = np.zeros((240, 320, 3), np.uint8)  # 帧
        self.aim = []  # aim 的格式是[flag， 类型， 颜色， 形状， (中心点坐标)， 面积]
        self.__cap = cv.VideoCapture(1)
        self.__cap.set(cv.CAP_PROP_FPS, 30)
        self.__cap.set(3, 320)  # 设置分辨率480p
        self.__cap.set(4, 240)
        self.box = []
        self.event = threading.Event()
        self.color_aim = "red"
        self.__kernel_circle = np.array(
            [
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 0],
            ],
            np.uint8,
        )

    def __capture(self):  # 拍照
        if not self.__cap.isOpened():
            print("Camera is not opened")
            return

        while True:
            ret, frame = self.__cap.read()
            self.event.wait()
            if ret:
                if self.aim != []:
                    self.box = self.aim[0]
                self.aim.clear()
                self.__frame = frame.copy()
                # cv.imshow("Vision", self.__frame)
                # cv.waitKey(1)
            else:
                print("Camera is not opened")

    def __fast_process(self):  # 快速处理
        while True:
            if self.__frame is None:
                continue

            self.event.wait()
            img = self.__frame.copy()  # 获取图像
            # 高斯模糊 和 HSV空间的转换
            gs_img = cv.GaussianBlur(img, (3, 3), 0)
            hsv = cv.cvtColor(gs_img, cv.COLOR_BGR2HSV)
            opening_hsv = hsv.copy()
            del gs_img, hsv

            if self.color_aim == "red":
                inRange_hsv_r = cv.inRange(
                    opening_hsv,
                    self.COLOR_THRESHOLD["red"]["Lower"],
                    self.COLOR_THRESHOLD["red"]["Upper"],
                )
                inRange_hsv_r_2 = cv.inRange(
                    opening_hsv,
                    self.COLOR_THRESHOLD["red2"]["Lower"],
                    self.COLOR_THRESHOLD["red2"]["Upper"],
                )
                inRange_hsv_r = cv.bitwise_or(inRange_hsv_r, inRange_hsv_r_2)
            else:
                inRange_hsv_b = cv.inRange(
                    opening_hsv,
                    self.COLOR_THRESHOLD["blue"]["Lower"],
                    self.COLOR_THRESHOLD["blue"]["Upper"],
                )
            inRange_hsv_y = cv.inRange(
                opening_hsv,
                self.COLOR_THRESHOLD["yellow"]["Lower"],
                self.COLOR_THRESHOLD["yellow"]["Upper"],
            )
            inRange_aim = cv.bitwise_or(inRange_hsv_r, inRange_hsv_y)

            # 轮廓检测
            contours = cv.findContours(
                inRange_aim, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )[0]
            del inRange_hsv_r, inRange_hsv_y, inRange_hsv_r_2, inRange_aim
            for contour in contours:
                # 获取轮廓位置
                # rect = cv.minAreaRect(contour)
                x, y, w, h = cv.boundingRect(contour)

                # 过滤条件
                if (
                    w * h < 2050  # 面积小于2000
                    or w * h > 60000  # 面积大于60000
                    or w < 10
                    or h < 10  # 宽度或高度小于20
                ):
                    continue

                # 颜色识别
                color_judge = {
                    "red": 0,
                    "blue": 0,
                    "yellow": 0,
                }

                step = 5  # 颜色识别采样的步长
                for i in range(x, x + w, step):
                    for j in range(y, y + h, step):
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
                color = max(color_judge, key=color_judge.get)

                # 结果输出
                # aim 的格式是[flag， 类型， 颜色， 形状， (中心点坐标)， 面积]
                box = [
                    True,
                    "normal",
                    color,
                    "NULL",
                    (x + w / 2, y + h / 2),
                    w * h,
                ]
                print(box)
                self.box = box

    def __normal_process(self):  # 图像处理
        while True:
            if self.__frame is None:
                continue

            self.event.wait()
            img = self.__frame.copy()  # 获取图像
            # 高斯模糊 和 HSV空间的转换
            gs_img = cv.GaussianBlur(img, (3, 3), 0)
            hsv = cv.cvtColor(gs_img, cv.COLOR_BGR2HSV)
            opening_hsv = hsv.copy()
            del gs_img, hsv

            # 颜色范围的掩膜,获取所需颜色的掩膜
            inRange_hsv_r = cv.inRange(
                opening_hsv,
                self.COLOR_THRESHOLD["red"]["Lower"],
                self.COLOR_THRESHOLD["red"]["Upper"],
            )
            inRange_hsv_b = cv.inRange(
                opening_hsv,
                self.COLOR_THRESHOLD["blue"]["Lower"],
                self.COLOR_THRESHOLD["blue"]["Upper"],
            )
            inRange_hsv_y = cv.inRange(
                opening_hsv,
                self.COLOR_THRESHOLD["yellow"]["Lower"],
                self.COLOR_THRESHOLD["yellow"]["Upper"],
            )
            inRange_hsv_r_2 = cv.inRange(
                opening_hsv,
                self.COLOR_THRESHOLD["red2"]["Lower"],
                self.COLOR_THRESHOLD["red2"]["Upper"],
            )
            inRange_hsv_r = cv.bitwise_or(inRange_hsv_r, inRange_hsv_r_2)

            # 合成目标二值化图像
            inRange_aim = cv.bitwise_or(inRange_hsv_r, inRange_hsv_b)
            inRange_aim = cv.bitwise_or(inRange_aim, inRange_hsv_y)

            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

            inRange_aim = cv.medianBlur(inRange_aim, 5)
            # 目标图像进行形态学处理
            inRange_aim = cv.morphologyEx(
                inRange_aim, cv.MORPH_CLOSE, self.__kernel_circle
            )
            # inRange_aim = cv.GaussianBlur(inRange_aim, (5, 5), 0)
            cv.imshow("inRange_aim", inRange_aim)
            # 轮廓检测
            contours = cv.findContours(
                inRange_aim.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )[0]
            del (
                inRange_hsv_r,
                inRange_hsv_b,
                inRange_hsv_y,
                inRange_hsv_r_2,
                inRange_aim,
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

                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv.putText(
                    img,
                    str(round(area_ratio, 2)),
                    (x + 80, y - 2),
                    cv.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

                # color = 'unknown'
                color_judge = {
                    "red": 0,
                    "blue": 0,
                    "yellow": 0,
                }

                step = 10  # 颜色识别采样的步长
                for i in range(x, x + w, step):
                    for j in range(y, y + h, step * 2):
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
                color = max(color_judge, key=color_judge.get)

                cv.putText(
                    img,
                    str(round(min(w, h) / max(w, h), 2)),
                    (x, y - 2),
                    cv.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

                # cv.putText(
                #     img,
                #     color,
                #     (x, y - 2),
                #     cv.FONT_HERSHEY_COMPLEX,
                #     0.5,
                #     (0, 0, 255),
                #     1,
                # )

                cv.circle(img, (int(rect[0][0]), int(rect[0][1])), 1, (0, 0, 0), -1)

                # 识别形状
                normal_shape = "NULL"
                approx = cv.approxPolyDP(
                    contour, 0.025 * cv.arcLength(contour, True), True
                )
                # if len(approx) >= 7 and int(area_ratio * 10) == 7:
                #     normal_shape = "circle"
                # else:
                #     normal_shape = "block"

                if len(approx) < 7 and int(area_ratio * 10) > 7:
                    normal_shape = "block"
                elif len(approx) >= 7 and int(area_ratio * 10) <= 7:
                    normal_shape = "circle"
                else:
                    normal_shape = "unknown"
                cv.putText(
                    img,
                    str(len(approx)),
                    (x + 60, y - 2),
                    cv.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

                # 结果输出
                # aim 的格式是[flag， 类型， 颜色， 形状， (中心点坐标)， 面积]
                box = [
                    True,
                    "normal",
                    color,
                    normal_shape,
                    (int(x + w / 2), int(y + h / 2)),
                    out_area,
                ]
                print("[Output]:", box)
                self.aim.append(box)
            cv.imshow("pro", img)
            cv.waitKey(1)

    def __QRcode_process(self):  # 二维码识别
        while True:
            if self.__frame is None:
                time.sleep(0.05)
                continue

            self.event.wait()
            # 获取图像
            QR_img = self.__frame.copy()
            QR_show = QR_img.copy()
            QR_img = cv.cvtColor(QR_img, cv.COLOR_BGR2GRAY)
            # QR_img = cv.equalizeHist(QR_img)

            for code in pyzbar.decode(QR_img):
                data = code.data.decode("utf-8")
                if data == "B":
                    QR_color = "blue"
                else:
                    QR_color = "red"

                cv.rectangle(
                    QR_show,
                    (code.rect[0], code.rect[1]),
                    (code.rect[0] + code.rect[2], code.rect[1] + code.rect[3]),
                    (0, 255, 0),
                    1,
                )
                cv.putText(
                    QR_show,
                    data,
                    (code.rect[0], code.rect[1]),
                    cv.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

                box = [
                    True,
                    "QRcode",
                    QR_color,
                    "block",
                    (
                        code.rect[0] + code.rect[2] // 2,
                        code.rect[1] + code.rect[3] // 2,
                    ),
                    code.rect[2] * code.rect[3],
                ]
                cv.imshow("QRcode", QR_show)
                self.aim.append(box)

    def start_control(self):  # 启动控制
        capture_threading = threading.Thread(target=self.__capture)
        normal_threading = threading.Thread(target=self.__normal_process)
        QRcode_threading = threading.Thread(target=self.__QRcode_process)
        fast_threading = threading.Thread(target=self.__fast_process)
        capture_threading.start()  # 启动拍照线程
        time.sleep(0.1)
        # fast_threading.start()  # 启动快速处理线程
        normal_threading.start()  # 启动图像处理线程
        # QRcode_threading.start()  # 启动图像处理线程
        self.event.set()

    def stop_control(self, chose=None):  # 停止控制
        print("Vision Error")
        pass


# v = Vision()
if __name__ == "__main__":
    vision = Vision()
    print("Vision Start")
    vision.start_control()
