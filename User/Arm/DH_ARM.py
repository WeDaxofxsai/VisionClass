from math import radians, degrees, cos, sin, sqrt, atan2, acos, pi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

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
        顺着机械臂看, x 向前, y 向右, z 向上
        """
        self.l1: float = 200.0
        self.l2: float = 200.0
        self.l3: float = 70.35
        self.ld: float = 71.5  # l down
        self.lu: float = 45.0  # l up

        self.theta1: float = radians(theta1)
        self.theta2: float = radians(theta2)
        self.theta3: float = 0
        self.angle: float = angle
        self.T04 = self.calculate_T()
        self.xyz: list = self.update()

    def _calculate_acos(self, l1: float, l2: float, l3: float) -> float:
        """
        计算l1与l2的夹角, l3为对边
        """
        # Check triangle validity
        if (l1 + l2 > l3) and (l1 + l3 > l2) and (l2 + l3 > l1):
            return acos((l1**2 + l2**2 - l3**2) / (2 * l1 * l2))
        else:
            raise ValueError("The given lengths cannot form a triangle.")

    def _radian_to_degree(self, radian: float) -> float:
        """
        将弧度转换为角度, 保留一位小数
        """
        return round(degrees(radian), 1)

    def _build_transformation_matrix(
        self, theta: float, a: float, alpha: float = 0, d: float = 0
    ) -> np.array:
        """
        构造DH传递矩阵
        """
        return np.array(
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

    def calculate_T(self) -> np.array:
        """
        计算末端位姿矩阵
        """
        # 连杆夹角计算
        kesi1: float = self.theta1
        l: float = sqrt(
            self.l1**2
            + self.ld**2
            - 2 * self.l1 * self.ld * cos(self.theta2 - self.theta1)
        )
        kesi2: float = (
            pi
            - self._calculate_acos(self.lu, l, self.l1)
            - self._calculate_acos(l, self.l1, self.ld)
        )
        kesi3: float = pi - kesi1 - kesi2

        T01 = self._build_transformation_matrix(self.angle, 0, pi / 2)
        T12 = self._build_transformation_matrix(kesi1, self.l1)
        T23 = self._build_transformation_matrix(kesi2 - pi, self.l2)
        T34 = self._build_transformation_matrix(kesi3, self.l3)

        return np.dot(np.dot(np.dot(T01, T12), T23), T34).astype(np.float16)

    def update(self) -> None:
        """
        更新末端位置
        """
        self.xyz = self.calculate_T()[0:3, 3]
        print(
            "[Update] "
            f"Now: angle = {self._radian_to_degree(self.angle)}, "
            f"theta1 = {self._radian_to_degree(self.theta1)} "
            f"theta2 = {self._radian_to_degree(self.theta2)} "
            f"theta3 = {self._radian_to_degree(self.theta3)} "
            f"arm end position: {self.xyz}"
        )

    def inverse_kinematics(self, x: float, y: float, z: float) -> None:
        """
        逆运动学求解各关节角度与变化量
        x y z: 目标末端位置
        """
        angle: float = atan2(y, x)  # 云台角度
        # Caculate kesis
        x = sqrt(x**2 + y**2) - self.l3
        L: float = sqrt(x**2 + z**2)
        kesi1: float = atan2(z, x) + self._calculate_acos(
            self.l1, L, self.l2
        )  # 主臂与上底板夹角
        kesi2: float = self._calculate_acos(self.l1, self.l2, L)  # 副臂与主臂夹角
        kesi3: float = pi - kesi1 - kesi2  # 前臂与副臂夹角

        # kesis to thetas
        theta1: float = kesi1
        l: float = sqrt(
            self.lu**2 + self.l1**2 - 2 * self.lu * self.l1 * cos(pi - kesi2)
        )
        theta2: float = (
            kesi1
            + self._calculate_acos(self.l1, l, self.lu)
            + self._calculate_acos(l, self.ld, self.l1)
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

        # Output
        print(
            "[Inverse_kinematics] "
            f"delta_angle = {self._radian_to_degree(delta_angle)}, "
            f"delta_theta1 = {self._radian_to_degree(delta_theta1)}, "
            f"delta_theta2 = {self._radian_to_degree(delta_theta2)}, "
            f"delta_theta3 = {self._radian_to_degree(delta_theta3)}"
        )
        self.update()
        return delta_angle, delta_theta1, delta_theta2, delta_theta3

    def trun(self, delta_angle: float):
        self.angle += radians(delta_angle)
        print(f"[Trun] delta_angle = {delta_angle} degree")
        self.update()

    def euler2rot(self, roll: float, pitch: float, yaw: float):
        """
        欧拉角转旋转矩阵
        """
        Rx = np.array(
            [[1, 0, 0], [0, cos(roll), -sin(roll)], [0, sin(roll), cos(roll)]]
        )
        Ry = np.array(
            [[cos(pitch), 0, sin(pitch)], [0, 1, 0], [-sin(pitch), 0, cos(pitch)]]
        )
        Rz = np.array([[cos(yaw), -sin(yaw), 0], [sin(yaw), cos(yaw), 0], [0, 0, 1]])
        return np.dot(Rz, np.dot(Ry, Rx))

    def cameraToObject(self, Zc: float, box: list) -> list:
        """
        将相机坐标系下的box坐标转换为世界坐标系下的box坐标
        box: [x, y]
        """
        c = Zc * np.array([box[0], box[1], 1])
        c.reshape(3, 1)
        Cmt = np.array(
            [
                [716.16568025, 0.0, 324.88785704],
                [0.0, 715.54284287, 207.04860251],
                [0.0, 0.0, 1.0],
            ]
        )
        Cmt = np.linalg.inv(Cmt)
        T = np.array([self.xyz[1], self.xyz[0], self.xyz[2]])
        T.reshape(3, 1)
        R = self.euler2rot(0, 0, self.angle)
        o = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])
        tao = np.dot(o, np.dot(R, (np.dot(Cmt, c) - T)))
        print(f"[cameraToObject] target postion: {tao}")
        return tao

    def plot_arm(self, target=None):
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
            pi
            - self._calculate_acos(self.lu, l, self.l1)
            - self._calculate_acos(l, self.l1, self.ld)
        )
        kesi3: float = pi - kesi1 - kesi2

        # 计算每个关节的位置
        T01 = self._build_transformation_matrix(self.angle, 0, pi / 2)
        T12 = self._build_transformation_matrix(kesi1, self.l1)
        T23 = self._build_transformation_matrix(kesi2 - pi, self.l2)
        T34 = self._build_transformation_matrix(kesi3, self.l3)

        # 计算关节位置
        P0 = np.array([0, 0, 0, 1])
        P1 = np.dot(T01, P0)
        P2 = np.dot(T01 @ T12, P0)
        P3 = np.dot(T01 @ T12 @ T23, P0)
        P4 = np.dot(T01 @ T12 @ T23 @ T34, P0)

        # 绘制机械臂的各个关节
        ax.plot(
            [P0[0], P1[0], P2[0], P3[0], P4[0]],
            [P0[1], P1[1], P2[1], P3[1], P4[1]],
            [P0[2], P1[2], P2[2], P3[2], P4[2]],
            "bo-",
            lw=5,
            markersize=5,  # 标记大小
            markeredgecolor="black",  # 标记边缘颜色
            markerfacecolor="red",  # 标记内部颜色
        )

        ax.plot([P4[0], P4[0]], [P4[1], P4[1]], [P4[2], 0], "b--", lw=0.5)

        # 如果提供了目标位置，则绘制目标
        if target is not None:
            ax.scatter(target[0], target[1], target[2], color="r", s=10)
            ax.text(target[0], target[1], target[2], "Target", color="red")

        # 设置坐标轴范围
        ax.set_xlim(-300, 500)
        ax.set_ylim(-300, 500)
        ax.set_zlim(0, 500)

        # 设置比例相同（仅视觉效果）
        max_range = max(500 - (-300), 500 - (-300), 500)
        ax.set_xlim([-max_range / 2, max_range / 2])
        ax.set_ylim([-max_range / 2, max_range / 2])
        ax.set_zlim([0, max_range / 2])

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Robot Arm")

        plt.show()


if __name__ == "__main__":
    # arm = Arm(90, 178.73)
    arm = Arm(143.30, 200.46)
    # l: float = sqrt(200**2 + 45.2**2)
    # print(90 + degrees(acos((70 ** 2 + l ** 2 - 200 ** 2) / (2 * l * 70)) + acos(200 / l)))
    delta_angle, delta_theta1, delta_theta2, delta_theta3 = arm.inverse_kinematics(
        300.0, 0.0, 300.0
    )
    arm.trun(30)
    tao = arm.cameraToObject(70, [11, 11])
    arm.plot_arm(tao)
