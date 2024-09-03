import numpy as np
import matplotlib.pyplot as plt

# 假设这里是你根据 DH 参数计算得到的各个关节的位姿矩阵
T_matrices = [np.eye(4), T1, T1 @ T2, T1 @ T2 @ T3, ...]  # 根据实际情况替换

# 提取各个关节的位置
positions = [T[:3, 3] for T in T_matrices]

# 分别获取 x, y, z 坐标
x = [pos[0] for pos in positions]
y = [pos[1] for pos in positions]
z = [pos[2] for pos in positions]

# 绘制机械臂
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(x, y, z, marker="o")

# 设置坐标轴标签
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# 设置图形的视角
ax.view_init(elev=30, azim=45)

plt.show()
