from math import radians, degrees, cos, sin, sqrt, atan2, acos, pi
import numpy as np
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D


class Arm:
    instance = None

    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)

        return cls.instance

    def __init__(self, theta1: float, theta2: float, angle: float = 0):
        """
        l1 / 2 / 3: 从下往上各臂长
        theta1 / 2 / 3: 从下往上3关节的旋转角度
        angle: 云台角度
        xyz: 末端位置
        顺着机械臂看, x 向前, y 向左, z 向上
        """
        self.l1: float = 200.0
        self.l2: float = 200.0
        self.l3: float = 70.35
        self.ld: float = 71.5  # l down
        self.lu: float = 45.0  # l up

        self.theta1: float = radians(theta1)
        self.theta2: float = radians(theta2)
        self.theta3: float = 0
        self.angle: float = radians(angle)
        self.xyz: np.ndarray = self.update()

    def __acos(self, l1: float, l2: float, l3: float) -> float:
        """
        计算l1与l2的夹角, l3为对边
        """
        # Check triangle validity
        if (l1 + l2 > l3) and (l1 + l3 > l2) and (l2 + l3 > l1):
            return acos((l1**2 + l2**2 - l3**2) / (2 * l1 * l2))
        else:
            raise ValueError("The given lengths cannot form a triangle.")

    def __rad2deg(self, radian: float) -> float:
        """
        将弧度转换为角度, 保留一位小数
        """
        return round(degrees(radian), 1)

    def buildTFMatrix(
        self, theta: float, a: float, alpha: float = 0, d: float = 0
    ) -> np.matrix:
        """
        构造DH传递矩阵
        """
        return np.matrix(
            [
                [
                    cos(theta),
                    -sin(theta) * cos(alpha),
                    sin(theta) * sin(alpha),
                    a * cos(theta),
                ],
                [
                    sin(theta),
                    cos(theta) * cos(alpha),
                    -cos(theta) * sin(alpha),
                    a * sin(theta),
                ],
                [0, sin(alpha), cos(alpha), d],
                [0, 0, 0, 1],
            ]
        )

    def calcEndMatrix(self) -> np.matrix:
        """
        计算末端位姿矩阵
        """
        # 连杆夹角计算
        kesi1: float = self.theta1
        _L: float = sqrt(
            self.l1**2
            + self.ld**2
            - 2 * self.l1 * self.ld * cos(self.theta2 - self.theta1)
        )
        kesi2: float = (
            pi - self.__acos(self.lu, _L, self.l1) - self.__acos(_L, self.l1, self.ld)
        )
        kesi3: float = pi - kesi1 - kesi2

        if self.theta3 == 0:
            self.theta3 = kesi3

        T01 = self.buildTFMatrix(self.angle, 0, pi / 2)
        T12 = self.buildTFMatrix(kesi1, self.l1)
        T23 = self.buildTFMatrix(kesi2 - pi, self.l2)
        T34 = self.buildTFMatrix(kesi3, self.l3)

        return np.dot(np.dot(np.dot(T01, T12), T23), T34).astype(np.float16)

    def update(self) -> None:
        """
        更新末端位置
        """
        self.xyz = self.calcEndMatrix()[0:3, 3].T.A1
        print(
            "[Update] "
            f"Now: angle = {self.__rad2deg(self.angle)}, "
            f"theta1 = {self.__rad2deg(self.theta1)} "
            f"theta2 = {self.__rad2deg(self.theta2)} "
            f"theta3 = {self.__rad2deg(self.theta3)} "
            f"arm end position: {self.xyz}"
        )

    def goToPosition(self, x: float, y: float, z: float) -> float:
        """
        逆运动学求解各关节角度与变化量
        x y z: 目标末端位置
        """
        if x == 0 and y == 0:
            raise ValueError("x y cannot both be zero")
        angle: float = atan2(y, x)  # 云台角度

        # Caculate kesis
        x = sqrt(x**2 + y**2) - self.l3
        L: float = sqrt(x**2 + z**2)
        kesi1: float = atan2(z, x) + self.__acos(
            self.l1, L, self.l2
        )  # 主臂与上底板夹角
        kesi2: float = self.__acos(self.l1, self.l2, L)  # 副臂与主臂夹角
        kesi3: float = pi - kesi1 - kesi2  # 前臂与副臂夹角

        # kesis to thetas
        theta1: float = kesi1
        L = sqrt(self.lu**2 + self.l1**2 - 2 * self.lu * self.l1 * cos(pi - kesi2))
        theta2: float = (
            kesi1 + self.__acos(self.l1, L, self.lu) + self.__acos(L, self.ld, self.l1)
        )
        theta3: float = kesi3

        # Update
        delta_angle = angle - self.angle
        delta_theta1 = theta1 - self.theta1
        delta_theta2 = theta2 - self.theta2
        delta_theta3 = theta3 - self.theta3

        self.angle = angle
        self.theta1 = theta1
        self.theta2 = theta2
        self.theta3 = theta3
        self.update()

        # Output
        print(
            "[goToPosition] "
            f"delta_angle = {self.__rad2deg(delta_angle)}, "
            f"delta_theta1 = {self.__rad2deg(delta_theta1)}, "
            f"delta_theta2 = {self.__rad2deg(delta_theta2)}, "
            f"delta_theta3 = {self.__rad2deg(delta_theta3)}"
        )

        return delta_angle, delta_theta1, delta_theta2, delta_theta3

    def trunToAngle(self, target_angle: float) -> None:
        print(f"[trunToAngle] delta_angle = {target_angle - self.angle} degree")
        self.angle = radians(target_angle)
        self.update()

    def euler2rot(self, roll: float, pitch: float, yaw: float) -> np.matrix:
        """
        欧拉角 -> 旋转矩阵
        """
        Rx = np.matrix(
            [
                [1, 0, 0],
                [0, cos(roll), -sin(roll)],
                [0, sin(roll), cos(roll)],
            ]
        )
        Ry = np.matrix(
            [
                [cos(pitch), 0, sin(pitch)],
                [0, 1, 0],
                [-sin(pitch), 0, cos(pitch)],
            ]
        )
        Rz = np.matrix(
            [
                [cos(yaw), -sin(yaw), 0],
                [sin(yaw), cos(yaw), 0],
                [0, 0, 1],
            ]
        )
        return np.dot(Rz @ Ry @ Rx, np.eye(3))

    def camera2object(self, Zc: float, box: list) -> np.ndarray:
        """
        将相机坐标系下的box坐标转换为世界坐标系下的box坐标
        box: [x, y]
        """
        if not (0 <= Zc <= self.xyz[2]) or not (
            0 <= box[0] <= 640 or 0 <= box[1] <= 480
        ):
            raise ValueError("Out of range")

        c = Zc * np.matrix([box[0], box[1], 1]).T
        Cmt = np.matrix(
            [  # 相机内参矩阵
                [716.16568025, 0.0, 324.88785704],
                [0.0, 715.54284287, 207.04860251],
                [0.0, 0.0, 1.0],
            ]
        )
        Tvec = np.matrix(self.xyz).T  # 平移向量
        Rmt = self.euler2rot(0, pi, pi / 2 + self.angle)  # 旋转矩阵
        print(self.__rad2deg(self.angle))

        tao = (np.dot(Rmt.I, np.dot(Cmt.I, c)) + Tvec).T.A1  # 目标点坐标
        print(f"[camera2object] target postion: {tao}")
        return tao

    def plotArm(self, targets: list = []):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # 连杆夹角计算
        kesi1: float = self.theta1
        l: float = sqrt(
            self.l1**2
            + self.ld**2
            - 2 * self.l1 * self.ld * cos(self.theta2 - self.theta1)
        )
        kesi2: float = (
            pi - self.__acos(self.lu, l, self.l1) - self.__acos(l, self.l1, self.ld)
        )
        kesi3: float = pi - kesi1 - kesi2

        # 计算每个关节的位置
        T01 = self.buildTFMatrix(self.angle, 0, pi / 2)
        T12 = self.buildTFMatrix(kesi1, self.l1)
        T23 = self.buildTFMatrix(kesi2 - pi, self.l2)
        T34 = self.buildTFMatrix(kesi3, self.l3)

        # 计算关节位置
        P0 = np.array([0, 0, 0, 1])
        P1 = np.dot(T01, P0).A1
        P2 = np.dot(T01 @ T12, P0).A1
        P3 = np.dot(T01 @ T12 @ T23, P0).A1
        P4 = np.dot(T01 @ T12 @ T23 @ T34, P0).A1

        # 绘制机械臂的各个关节
        ax.plot(
            [P0[0], P1[0], P2[0], P3[0], P4[0]],
            [P0[1], P1[1], P2[1], P3[1], P4[1]],
            [P0[2], P1[2], P2[2], P3[2], P4[2]],
            "bo-",
        )

        # 如果提供了目标位置，则绘制目标
        for target in targets:
            ax.scatter(target[0], target[1], target[2], color="r", s=10)
            ax.text(target[0], target[1], target[2], "tar", color="red")

        # 设置坐标轴范围
        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 500)
        ax.set_zlim(0, 400)

        # 设置比例相同（仅视觉效果）
        max_range = max(500 - (-500), 500 - (-500), 400 - 0)
        ax.set_xlim([-max_range / 2, max_range / 2])
        ax.set_ylim([-max_range / 2, max_range / 2])
        ax.set_zlim([0, max_range])

        # ylim = ax.get_ylim()
        # ax.set_ylim(ylim[::-1])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Robot Arm")

        plt.show()


if __name__ == "__main__":
    arm = Arm(143.30, 200.46)
    delta_angle, delta_theta1, delta_theta2, delta_theta3 = arm.goToPosition(
        300.0, 0.0, 300.0
    )
    for i in range(0, -91, -30):
        arm.trunToAngle(i)
        tao1 = arm.camera2object(70, [0, 0])
        # tao2 = arm.camera2object(70, [320, 240])
        arm.plotArm([tao1])
