import cv2 as cv
import numpy as np
import time

# 70.34685516   0.         210.8011322 ], Upper = [130.34686279  42.63784027 255
COLOR_THRESHOLD = {
    "white": {
        "Lower": np.array([70, 0, 210]),
        "Upper": np.array([130, 42, 255]),
    },
    "red": {
        "Lower": np.array([0, 43, 46]),
        "Upper": np.array([10, 255, 255]),
    },
    "red2": {
        "Lower": np.array([156, 43, 46]),
        "Upper": np.array([180, 255, 255]),
    },
    "blue": {
        "Lower": np.array([85, 46, 221]),
        "Upper": np.array([124, 255, 255]),
    },
    "yellow": {
        "Lower": np.array([24, 72, 60]),
        "Upper": np.array([35, 255, 255]),
    },
}
kernel_circle = np.array(
    [
        [0, 1, 1, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 1, 1, 0],
    ],
    np.uint8,
)

# image = cv.imread(r"E:\project\User\Things\white_ball.jpg")
# cv.imshow("Original Image", image)

# capture = cv.VideoCapture(1)
# capture.set(cv.CAP_PROP_FRAME_WIDTH, 320)
# capture.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
# while True:
#     ret, image = capture.read()
#     cv.imshow("Video", image)

#     # start_time = time.time()
#     hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
#     inRange_hsv = cv.inRange(
#         hsv,
#         COLOR_THRESHOLD["white"]["Lower"],
#         COLOR_THRESHOLD["white"]["Upper"],
#     )
#     cv.imshow("White Ball", inRange_hsv)

#     inRange_aim = cv.morphologyEx(inRange_hsv, cv.MORPH_CLOSE, kernel_circle)
#     cv.imshow("Aim", inRange_aim)
#     inRange_aim = cv.medianBlur(inRange_hsv, 5)
#     imgray = cv.Canny(inRange_aim, 30, 100)
#     cv.imshow("Aim_", imgray)

#     circles = cv.HoughCircles(
#         imgray,
#         cv.HOUGH_GRADIENT,
#         2,
#         80,
#         param1=100,
#         param2=100,
#         # minRadius=40,
#         # maxRadius=150,
#     )
#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for i in circles[0, :]:
#             # 画图
#             cv.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
#             cv.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
#             cv.putText(
#                 image,
#                 "center",
#                 (i[0] - 20, i[1] - 20),
#                 cv.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (255, 255, 255),
#                 2,
#             )
#             cv.putText(
#                 image,
#                 "radius",
#                 (i[0] - 20, i[1] + 20),
#                 cv.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (255, 255, 255),
#                 2,
#             )
#             cv.putText(
#                 image,
#                 str(i[2]),
#                 (i[0] + 20, i[1] + 20),
#                 cv.FONT_HERSHEY_SIMPLEX,
#                 0.5,
#                 (255, 255, 255),
#                 2,
#             )
#             print(i[0], i[1], i[2])
#     else:
#         print("No circles were found")
#     # circles = np.uint16(np.around(circles))

#     # print("--- %s seconds ---" % (time.time() - start_time))
#     cv.imshow("Original Image", image)
#     cv.waitKey(1)


# video_cap = cv.VideoCapture(r"E:/project/User/Things/white_ball.mp4")
# print("okk")
# cap = cv.VideoCapture("E:/project/User/Things/white_ball.mp4")  # 读取视频
# while cap.isOpened():
#     ret, frame = cap.read()
#     if ret:
#         cv.imshow("frame", frame)
#         key = cv.waitKey(25)
#         if key == ord("q"):
#             cap.release()
#             break
#     else:
#         cap.release()
# cv.destroyAllWindows()
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


print(softmax(np.asarray([2.0, 1.0, 0.1])))
