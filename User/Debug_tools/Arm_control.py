import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 读取图片
img1 = mpimg.imread("E:/project/User/Things/mixing.jpg")
img2 = mpimg.imread("E:/project/User/Things/map_.original.jpg")

# 创建一个新的图形
plt.figure()

# 添加第一个子图
plt.subplot(1, 2, 1)  # (rows, columns, panel number)
plt.imshow(img1)
plt.title("Image 1")
plt.axis("off")  # 关闭坐标轴

# 添加第二个子图
plt.subplot(1, 2, 2)
plt.imshow(img2)
plt.title("Image 2")
plt.axis("off")

# 显示图形
plt.show()
