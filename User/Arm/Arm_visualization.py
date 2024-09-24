# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import threading
import time
import matplotlib.pyplot as plt
from DH_ARM import Arm
import math


"""
1 创建画布 其中有两个子图
2 定义鼠标点击 按键点击等事件的回调函数
3 描述这一过程
    3.1 画布建立 子图建立
        3.1.1 画布包含的是两个子图
    3.2 创建点的对象
    3.3 创建交互的回调函数
    3.4 连接这些事件
    3.5 显示图形
    3.6 开始交互
        3.6.1 鼠标点击
        3.6.2 查找鼠标对应的点 依靠距离判断
        3.6.3 更新点的状态 是否被选中 选中的点有且只有一个
        3.6.4 鼠标的移动 更新被选中的点 这个过程是边移动边更新
        3.6.5 更新画布
        3.6.6 鼠标释放 取消选中点 记录这个刻点的状态
"""


import matplotlib.pyplot as plt


class JointPoint:
    def __init__(self, _x=0, _y=0, _z=0):
        self.x = _x
        self.y = _y
        self.z = _z
        self.state = False

    def update_point(self, _x: float, _y: float, _z: float):
        self.x = _x
        self.y = _y
        self.z = _z

    def set_state(self, _state: bool):
        self.state = _state


class Canvas:
    def __init__(self, _points: list = []):
        self.fig, self.axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6), dpi=100)
        self.points = _points
        self.selected_point = None
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.arm = Arm(143.30, 200.46)
        self.axes_chose = None

    def start(self):
        P = self.arm.getPoints()
        for point in P:
            self.points.append(JointPoint(point[0], point[1], point[2]))
        # 子图一的设置
        self.axes[0].set_title("overlook")
        self.axes[0].set_xlim((0, 380))
        self.axes[0].set_ylim((-450, 450))

        self.axes[0].set_xlabel("X")
        self.axes[0].set_ylabel("Y")
        self.axes[0].set_aspect(1, adjustable="box")
        self.axes[0].plot(
            [self.points[0].x, self.points[-1].x],
            [self.points[0].y, self.points[-1].y],
            "bo-",
            label="ax1",
            linewidth=2,
        )
        # self.axes[0].plot(self.points[-1].x, self.points[-1].y, "bo")

        # 子图2的设置
        self.axes[1].set_title("side viewer")
        self.axes[1].set_xlim((-10, 500))
        self.axes[1].set_ylim((-10, 380))
        self.axes[1].set_xlabel("X`")
        self.axes[1].set_ylabel("Z")
        self.axes[1].set_aspect(1, adjustable="box")

        x_bar = []
        for i in range(5):
            x_bar.append(math.sqrt((self.points[i].x) ** 2 + (self.points[i].y) ** 2))
        z_bar = []
        for i in range(5):
            z_bar.append(self.points[i].z)
        self.axes[1].plot(
            x_bar,
            z_bar,
            "bo-",
            label="ax1",
            linewidth=2,
        )

        plt.show()

    def update_plot(self):
        self.axes[0].clear()  # 清除当前图像
        self.axes[0].set_title("overlook")
        self.axes[0].set_xlabel("Y")
        self.axes[0].set_ylabel("X")
        self.axes[0].set_xlim((0, 380))
        self.axes[0].set_ylim((-450, 450))
        # 重新绘制所有点

        self.axes[0].plot(
            [self.points[0].x, self.points[-1].x],
            [self.points[0].y, self.points[-1].y],
            "bo-",
            label="ax1",
            linewidth=2,
        )

        # 子图2的设置
        self.axes[1].clear()  # 清除当前图像
        self.axes[1].set_title("side viewer")
        self.axes[1].set_xlim((-500, 500))
        self.axes[1].set_ylim((-10, 380))
        self.axes[1].set_xlabel("X`")
        self.axes[1].set_ylabel("Z")
        self.axes[1].set_aspect(1, adjustable="box")

        x_bar = []
        for i in range(5):
            x_bar.append(math.sqrt((self.points[i].x) ** 2 + (self.points[i].y) ** 2))
        z_bar = []
        for i in range(5):
            z_bar.append(self.points[i].z)
        self.axes[1].plot(
            x_bar,
            z_bar,
            "bo-",
            label="ax1",
            linewidth=2,
        )
        plt.draw()  # 更新图像

    def on_press(self, event):
        # 检查是否在某个axes内
        print("Mouse click:", event.xdata, event.ydata)
        if event.inaxes == self.axes[0] and event.ydata is not None:
            self.axes_chose = 0
            # print("Mouse click:", event.xdata, event.ydata)
            for i, point in enumerate(self.points):
                # 检查鼠标点击的是否接近某个点（通过距离判断）
                distance = (
                    (point.x - event.xdata) ** 2 + (point.y - event.ydata) ** 2
                ) ** 0.5
                if distance < 20:  # 假设点的半径为20，可以根据需要调整
                    self.selected_point = i
                    # print(f"Selected point: {i}")
                    break
        elif event.inaxes == self.axes[1] and event.ydata is not None:
            # print("second subplot")
            self.axes_chose = 1
            # print("Mouse click:", event.xdata, event.ydata)
            for i, point in enumerate(self.points):
                # 检查鼠标点击的是否接近某个点（通过距离判断）
                distance = (
                    ((point.x**2 + point.y**2) ** 0.5 - event.xdata) ** 2
                    + (point.z - event.ydata) ** 2
                ) ** 0.5
                if distance < 20:  # 假设点的半径为20，可以根据需要调整
                    self.selected_point = i
                    # print(f"Selected point: {i}")
                    break

    def on_motion(self, event):
        if (
            self.selected_point is not None
            and event.inaxes == self.axes[0]
            and event.ydata is not None
        ):
            # 更新被选中点的位置
            self.points[self.selected_point].x = event.xdata
            self.points[self.selected_point].y = event.ydata
            # 重新绘制图形
            # print(f"Update point: {self.selected_point}")
            self.update_joint()
            self.update_plot()
        elif (
            event.ydata is not None
            and self.selected_point is not None
            and event.inaxes == self.axes[1]
        ):
            self.points[self.selected_point].z = event.ydata
            self.points[self.selected_point].x = event.xdata * (
                self.points[self.selected_point].x
                / (
                    (
                        self.points[self.selected_point].x ** 2
                        + self.points[self.selected_point].y ** 2
                    )
                    ** 0.5
                )
            )
            self.points[self.selected_point].y = event.xdata * (
                self.points[self.selected_point].y
                / (
                    (
                        self.points[self.selected_point].x ** 2
                        + self.points[self.selected_point].y ** 2
                    )
                    ** 0.5
                )
            )
            self.update_joint()
            self.update_plot()

    # 定义鼠标释放事件的回调函数
    def on_release(self, event):
        # if self.selected_point is not None:
        #     # print(
        #     #     "Update point:",
        #     #     self.points[self.selected_point].x,
        #     #     self.points[self.selected_point].y,
        #     # )
        self.selected_point = None  # 释放鼠标，取消选中点
        print(self.points[-1].x, self.points[-1].y, self.points[-1].z)

    def update_joint(self):
        if self.points != None:
            extremity_point = self.points[-1]
            _ = self.arm.goToPosition(
                extremity_point.x, extremity_point.y, extremity_point.z
            )
            _points = self.arm.getPoints()
            for i, point in enumerate(_points):
                self.points[i].update_point(point[0], point[1], point[2])


if __name__ == "__main__":
    # arm = Arm(143.30, 200.46)
    # delta_angle, delta_theta1, delta_theta2, delta_theta3 = arm.goToPosition(
    #     150.0, 0.0, 350.0
    # )
    # points = arm.get_points()
    # points = [JointPoint(100, 200, 300), JointPoint(400, 250, 60)]
    canvas = Canvas()
    delta_angle, delta_theta1, delta_theta2, delta_theta3 = canvas.arm.goToPosition(
        285.0, 0.0, 232.0
    )
    canvas.start()
