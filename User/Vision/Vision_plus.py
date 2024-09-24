import cv2 as cv
import numpy as np


class Vision:
    def __init__(self):
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
                "Lower": np.array([100, 43, 46]),
                "Upper": np.array([124, 255, 255]),
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
        self.__width = 320
        self.__height = 240
        self.__frame = np.zeros(
            (self.__height, self.__width, 3), np.uint8
        )  # 帧的初始化
        self.event = threading.Event()
        self.__cap = cv.VideoCapture(1)
        self.__cap.set(cv.CAP_PROP_FPS, 30)
        self.__cap.set(3, self.__width)  # 设置分辨率480p
        self.__cap.set(4, self.__height)  # 设置分辨率480p
        self.aim = []
        self.__kernel_circle = np.array(
            [
                [0, 1, 1, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                [0, 1, 1, 0],
            ],
            np.uint8,
        )
        self.__color = "red"

    def __capture(self):
        if not self.__cap.isOpened():
            print("[ERROR] Camera is not opened")
            return

        while True:
            self.event.wait()
            ret, frame = self.__cap.read()
            if ret:
                if self.aim != []:
                    self.box = self.aim[0]
                self.aim.clear()
            else:
                print("Camera is not opened")

    def show_image(self):
        pass

    def fast_detect(self):
        pass

    def detect(self):
        pass

    def start_control(self):
        pass


if __name__ == "__main__":
    vision = Vision()
    vision.start_control()
