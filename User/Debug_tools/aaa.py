import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


# 创建BCC晶胞的顶点和边信息
def create_bcc_lattice():
    # 立方体顶点坐标
    cube_vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )
    # BCC中心点
    center = np.array([[0.5, 0.5, 0.5]])

    # 返回顶点和中心点
    return np.vstack((cube_vertices, center))


# 添加立方体的连线
def draw_bcc_lines(ax, bcc_points):
    # 立方体边的连线（从立方体顶点相连）
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]

    # 连接中心点和各顶点的连线
    center_lines = [(8, i) for i in range(8)]  # 中心点的索引是8

    # 绘制所有边
    for edge in edges:
        ax.plot(
            [bcc_points[edge[0], 0], bcc_points[edge[1], 0]],
            [bcc_points[edge[0], 1], bcc_points[edge[1], 1]],
            [bcc_points[edge[0], 2], bcc_points[edge[1], 2]],
            color="b",
        )

    # 绘制中心点与顶点的连线
    for line in center_lines:
        ax.plot(
            [bcc_points[line[0], 0], bcc_points[line[1], 0]],
            [bcc_points[line[0], 1], bcc_points[line[1], 1]],
            [bcc_points[line[0], 2], bcc_points[line[1], 2]],
            color="r",
            linestyle="--",
        )


# 初始化3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# 获取晶胞点坐标
bcc_points = create_bcc_lattice()

# 设置图形的初始状态
scatter = ax.scatter(bcc_points[:, 0], bcc_points[:, 1], bcc_points[:, 2], s=100)

# 添加连线
draw_bcc_lines(ax, bcc_points)

# 设置轴的范围
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])


# 动画函数，更新每一帧的视角
def update(frame):
    ax.view_init(elev=20, azim=frame)
    return (scatter,)


# 创建动画，帧数为360，对应旋转360度
ani = FuncAnimation(fig, update, frames=360, interval=50)

# 保存动画到文件
# ani.save("bcc_lattice_with_lines.mp4", writer="ffmpeg", dpi=300)

plt.show()
time.sleep(10)
