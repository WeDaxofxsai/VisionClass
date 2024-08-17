# 二维码识别
import cv2
import numpy as np
from pyzbar.pyzbar import decode


# 读取文件
def Read():
    global Data_Array
    Data_Array = open('Authorited.txt').read().splitlines()  # 按行分隔
    print('已授权的数据：', Data_Array, '\n')


# 判断二维码是否授权
def Judge(data):
    global color
    if data in Data_Array:  # 成功
        color = (0, 255, 0)  # 绿色标记
        print('Authorized\n')
    else:  # 失败
        color = (0, 0, 255)  # 红色标记
        print('Unauthorized\n')


# 检测图像中的码（解码）
def Read_Decode_Pic(image):
    # 遍历解码
    for code in decode(image):
        # print("条形码/二维码：", code)
        data = code.data.decode('utf-8')
        print("条形码/二维码数据：", data)  # 解码数据

        # 判断二维码是否授权
        Judge(data)

        # 多边形获取（矩形的框）
        pts_poly = np.array(code.polygon, np.int32)  # 获取多边形坐标
        cv2.polylines(image, [pts_poly], True, color, 1)  # 画多边形框

        # 显示数据（获取矩形框的左上角作为Text的坐标(左边坐标)，显示数据）
        pts_rect = code.rect
        # print(pts_rect)
        cv2.putText(image, data, (pts_rect[0], pts_rect[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        #                  显示数据  矩形坐标                   字体类型                 字体大小 颜色     粗细

    cv2.imshow('image', image)  # 等画出所有矩形后显示


# 检测视频中的码（解码）
def Read_Decode_Cam():
    cap = cv2.VideoCapture(1)  # 打开视频
    cap.set(3, 320)  # 帧的宽度
    cap.set(4, 240)  # 帧的高度

    while True:
        success, image = cap.read()  # 获取每一帧图片
        cv2.imshow('image', image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图
        # image = cv2.equalizeHist(image)
        Read_Decode_Pic(image)  # 对每一帧图片检测

        cv2.waitKey(1)  # 延时1ms


if __name__ == '__main__':
    Read()  # 读取文件
    img = cv2.imread(r"E:\ultimately\Vision\Things\red&blue.jpg")
    # Read_Decode_Pic(img)  # 检测图像中的码（解码）
    Read_Decode_Cam()  # 检测视频中的码（解码）

    cv2.waitKey(0)