from Vision import Vision
import time
from Communication import Frame
import serial

v = Vision()
v.start_control()

ser = serial.Serial("/dev/ttyUSB0", 512000, timeout=0.5)
f = Frame(ser)
# f.call(box[2])
cnt = 0
while True:
    # time.sleep(0.005)
    print("p")
    box = v.box
    if (
        box != []
        and (box[2] == "red" or box[2] == "yellow")
        and box[4][0] > 250
        and box[4][0] < 290
    ):
        v.event.clear()
        print("[V]", box)
        f.call(box[2])
        v.event.set()
        v.box.clear()
        # cv.imshow("frame", v.__frame)
        time.sleep(0.6)
        cnt += 1
print("Done")
