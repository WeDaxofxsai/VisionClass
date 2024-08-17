from Vision import Vision
import time
from Communication import Frame
import serial

v = Vision()
v.start_control()

ser = serial.Serial("/dev/ttyUSB0", 512000, timeout=0.5)
f = Frame(ser)
while True:
    print("p")
    box = v.box[2]
    if box != [] and (box[2] == "red" or box[2] == "yellow") and box[4][0] > 275:
        v.event.wait()
        print("[V]", box)
        f.call(box[2])
        v.event.set()
        v.box.clear()
        time.sleep(0.3)
