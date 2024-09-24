import matplotlib.pyplot as plt
import numpy as np

# 生成 x 值
x = np.linspace(-10, 10, 400)

# 计算 arctan(x) 的 y 值
y = np.arctan(x)

# 绘制图像
plt.plot(x, y, label="arctan(x)")
plt.title("Plot of arctan(x)")
plt.xlabel("x")
plt.ylabel("arctan(x)")
plt.legend()
plt.grid(True)
plt.show()
