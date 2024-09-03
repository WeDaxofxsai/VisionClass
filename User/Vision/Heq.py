import cv2
import numpy as np

# 1. 读取图像并转换为HSV颜色空间
image = cv2.imread("E:/project/User/Things/red&blue.jpg")  # 替换为您的图像路径
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 2. 分离HSV通道
h, s, v = cv2.split(hsv_image)

# 3. 对V通道进行自适应阈值处理
v_adaptive_thresh = cv2.adaptiveThreshold(
    v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
)

# 4. 将处理后的V通道与原始的H和S通道合并
hsv_image_thresh = cv2.merge([h, s, v_adaptive_thresh])

# 5. 将结果从HSV转换回BGR颜色空间
result_image = cv2.cvtColor(hsv_image_thresh, cv2.COLOR_HSV2BGR)

# 6. 显示原始图像和自适应阈值处理后的图像
cv2.imshow("Original Image", image)
cv2.imshow("Adaptive Threshold in V Channel", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
