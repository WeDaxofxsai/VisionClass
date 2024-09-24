import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


# 创建BCC晶胞的顶点和边
def create_bcc_primitive_cell():
    a1 = np.array([0.5, -0.5, 0.5])
    a2 = np.array([0.5, 0.5, -0.5])
    a3 = np.array([-0.5, 0.5, 0.5])
    origin = np.array([0, 0, 0])
    vertices = np.array([origin, a1, a2, a3, a1 + a2, a2 + a3, a3 + a1, a1 + a2 + a3])
    return vertices


def create_bcc_unit_cell():
    cube_vertices = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
    )
    center = np.array([[0.5, 0.5, 0.5]])
    return np.vstack((cube_vertices, center))


def draw_bcc_primitive_cell(ax, cell_points):
    edges = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 6),
        (2, 4),
        (2, 5),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]
    for edge in edges:
        ax.plot(
            [cell_points[edge[0], 0], cell_points[edge[1], 0]],
            [cell_points[edge[0], 1], cell_points[edge[1], 1]],
            [cell_points[edge[0], 2], cell_points[edge[1], 2]],
            color="b",
            linestyle="--",
        )


def draw_bcc_unit_cell(ax, bcc_points):
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    center_lines = [(8, i) for i in range(8)]
    for edge in edges:
        ax.plot(
            [bcc_points[edge[0], 0], bcc_points[edge[1], 0]],
            [bcc_points[edge[0], 1], bcc_points[edge[1], 1]],
            [bcc_points[edge[0], 2], bcc_points[edge[1], 2]],
            color="black",
        )
    for line in center_lines:
        ax.plot(
            [bcc_points[line[0], 0], bcc_points[line[1], 0]],
            [bcc_points[line[0], 1], bcc_points[line[1], 1]],
            [bcc_points[line[0], 2], bcc_points[line[1], 2]],
            color="r",
            linestyle="--",
        )


def update(num, ax, bcc_primitive_points, bcc_unit_cell_points):
    ax.cla()  # Clear the current axes
    # 绘制BCC原胞
    ax.scatter(
        bcc_unit_cell_points[:, 0],
        bcc_unit_cell_points[:, 1],
        bcc_unit_cell_points[:, 2],
        s=100,
        c="k",
    )
    draw_bcc_unit_cell(ax, bcc_unit_cell_points)

    # 绘制BCC晶胞
    ax.scatter(
        bcc_primitive_points[:, 0],
        bcc_primitive_points[:, 1],
        bcc_primitive_points[:, 2],
        s=100,
        c="r",
    )
    draw_bcc_primitive_cell(ax, bcc_primitive_points)

    # 设置轴的范围和标签
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    ax.set_zlim([-0.5, 1.5])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("BCC Unit Cell (Green) and Primitive Cell (Blue)")
    ax.view_init(elev=10.0, azim=num)


# 初始化3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

bcc_primitive_points = create_bcc_primitive_cell()
bcc_unit_cell_points = create_bcc_unit_cell()

# 创建动画
ani = animation.FuncAnimation(
    fig,
    update,
    frames=360,  # np.arange(0, 360, 1),
    fargs=(ax, bcc_primitive_points, bcc_unit_cell_points),
    interval=50,
)

plt.show()
