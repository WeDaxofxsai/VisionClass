import cv2 as cv
import numpy as np
import time

# 读取图像并转换颜色空间
image = cv.imread("E:/project/User/Things/white_ball.jpg")  # 替换为您的图像路径
# image = cv.resize(image, (128, 96))  # 可选，调整图像大小
now_time = time.time()
hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# 转换为2D数组
pixel_values = hsv_image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# K-means聚类参数
k = 7  # 假设分为六个类
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels, centers = cv.kmeans(
    pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS
)
print(time.time() - now_time)
# 输出每个类的中心颜色
print("Cluster centers (HSV):")
for i, center in enumerate(centers):
    print(f"Cluster {i}: HSV = {center}")


# 计算每个类的HSV范围
def get_hsv_range(center, pixel_values, threshold=30):
    """
    根据聚类中心和像素值计算每个类的HSV范围。
    :param center: 聚类中心的HSV值
    :param pixel_values: 图像的HSV像素值
    :param threshold: 范围内的最大距离
    :return: 颜色范围的下界和上界
    """
    lower_bound = center - threshold
    upper_bound = center + threshold

    # 限制范围在有效HSV值范围内
    lower_bound = np.clip(lower_bound, [0, 0, 0], [179, 255, 255])
    upper_bound = np.clip(upper_bound, [0, 0, 0], [179, 255, 255])

    return lower_bound, upper_bound


print("\nHSV Ranges for each cluster:")
for i, center in enumerate(centers):
    lower_bound, upper_bound = get_hsv_range(center, pixel_values)
    print(f"Cluster {i}: Lower = {lower_bound}, Upper = {upper_bound}")

# 创建掩膜并显示每个聚类的图像
for i in range(k):
    # 创建掩膜，尺寸与原图像一致
    mask = (labels == i).astype(np.uint8)
    mask = mask.reshape(hsv_image.shape[:2])  # 重塑掩膜为原图像的大小
    mask = mask * 255  # 将掩膜值从 0 和 1 扩展到 0 和 255

    # 应用掩膜提取图像中的对应区域
    masked_image = cv.bitwise_and(image, image, mask=mask)

    # 显示每个类的图像
    cv.imshow(f"Cluster {i}", masked_image)

# 显示 K-means 分割结果
result = cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)
cv.imshow("Segmented Image", result)

cv.waitKey(0)
cv.destroyAllWindows()
