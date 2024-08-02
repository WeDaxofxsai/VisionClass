import cv2
import numpy as np


def horn_schunck_optical_flow(prev_frame, next_frame, alpha=1.0, iterations=5):
    # 转换为灰度图像
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # 设置Horn-Schunck参数
    params = {
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 15,
        'iterations': iterations,
        'poly_n': 5,
        'poly_sigma': 1.1,
        'flags': 0
    }

    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, **params)

    # 可视化光流
    flow_image = draw_optical_flow_arrows(prev_frame, flow)

    return flow_image


def draw_optical_flow_arrows(frame, flow, step=16):
    h, w = flow.shape[:2]
    y, x = np.mgrid[step // 2:h:step, step // 2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    # 创建线的终点坐标
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    # 创建光流可视化图像
    flow_image = np.copy(frame)
    cv2.polylines(flow_image, lines, 0, (0, 255, 0), 2)

    return flow_image


# 读取视频
video_cap = cv2.VideoCapture(r"E:/VisionClass/VisionClass/Things/output.mp4")
# 视频输出
fps = video_cap.get(cv2.CAP_PROP_FPS)  # 获得视频帧率，即每秒多少帧
size = (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
videoWriter = cv2.VideoWriter('./video/new_green.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)

ret, prev_frame = video_cap.read()

while True:
    # 读取下一帧
    ret, next_frame = video_cap.read()

    if not ret:
        break
    # 获取光流图像
    flow_image = horn_schunck_optical_flow(prev_frame, next_frame)

    # 显示结果
    cv2.imshow("Optical Flow (Horn-Schunck)", flow_image)

    # 更新上一帧
    prev_frame = next_frame

    # 退出条件
    if cv2.waitKey(30) & 0xFF == 27:
        break

# 释放资源
video_cap.release()
cv2.destroyAllWindows()