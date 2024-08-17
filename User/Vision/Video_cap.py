import cv2

# 初始化摄像头
cap = cv2.VideoCapture(1)  # 0 通常是默认摄像头的标识

# 定义编解码器并创建VideoWriter对象
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('../Things/output.mp4', fourcc, 30.0, (640, 480))

# 捕获视频帧
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("无法获取帧，结束录制")
        break

    # 写入帧到文件
    out.write(frame)

    # 显示帧
    # cv2.imshow('frame', frame)

    # 按 'q' 退出循环
    if cv2.waitKey(1) == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()