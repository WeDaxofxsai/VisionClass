import cv2 as cv

cap = cv.VideoCapture(1, cv.CAP_DSHOW)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 240)
flag = cap.isOpened()
print(flag)
index = "check_board"
img_cnt = 0
while flag:
    ret, frame = cap.read()
    cv.imshow("Capture_Paizhao", frame)
    k = cv.waitKey(1) & 0xFF
    if k == ord("s"):  # 按下s键，进入下面的保存图片操作
        cv.imwrite(
            "E://project//User//Things//Camera_calibration//"
            + index
            + str(img_cnt)
            + ".jpg",
            frame,
        )
        img_cnt += 1
        print("save" + str(index) + str(img_cnt) + ".jpg successfuly!")
        print("-------------------------")
    elif k == ord("q"):  # 按下q键，程序退出
        break
cap.release()  # 释放摄像头
cv.destroyAllWindows()  # 释放并销毁窗
