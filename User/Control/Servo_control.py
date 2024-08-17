import serial
import pigpio
import time
import os

FRAME_HEADER = 0x55  # 帧头
CMD_SERVO_MOVE = 0x03  # 舵机移动指令
CMD_ACTION_GROUP_RUN = 0x06  # 运行动作组指令
CMD_ACTION_GROUP_STOP = 0x07  # 停止动作组指令
CMD_ACTION_GROUP_SPEED = 0x0B  # 设置动作组运行速度
CMD_GET_BATTERY_VOLTAGE = 0x0F  # 获取电池电压指令

open_io = "sudo pigpiod"
os.system(open_io)
time.sleep(1)
pi = pigpio.pi()  # 初始化 pigpio库
ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=0.5)


def moveServo(id=None, position=None, time=None):
    buf = bytearray(b"\x55\x55")
    buf.append(8)
    buf.append(3)
    buf.append(1)
    buf.extend([(0xFF & time), (0xFF & (time >> 8))])  # 分低8位 高8位 放入缓存
    buf.append(id)
    buf.extend([(0xFF & position), (0xFF & (position >> 8))])
    ser.write(buf)  # 发送


# Function:     moveServos
# Description： 控制多个舵机转动
# Parameters:   Num:舵机个数,time:转动时间,*List:舵机ID,转动角，舵机ID,转动角度 如此类推
# Return:       无返回
def moveServos(Num=None, time=None, *List):
    buf = bytearray(b"\x55\x55")
    buf.append(Num * 3 + 5)
    buf.append(3)
    buf.append(Num)
    buf.extend([(0xFF & time), (0xFF & (time >> 8))])  # 分低8位 高8位 放入缓存
    for i in range(0, len(List) - 1, 2):
        id = List[i]
        position = List[i + 1]
        buf.append(0xFF & id)
        buf.extend(
            [(0xFF & position), (0xFF & (position >> 8))]
        )  # 分低8位 高8位 放入缓存
    ser.write(buf)  # 发送


# Function:     runActionGroup
# Description： 运行指定动作组
# Parameters:   NumOfAction:动作组序号, Times:执行次数
# Return:       无返回
# Others:       Times = 0 时无限循环
def runActionGroup(ID=None, time=None):
    buf = bytearray(b"\x55\x55")
    buf.append(5)
    buf.append(6)
    buf.append(ID)
    buf.extend([(0xFF & time), (0xFF & (time >> 8))])  # 分低8位 高8位 放入缓存
    ser.write(buf)  # 发送


if __name__ == "__main__":
    while True:
        print(1)
        moveServo(1, 300, 500)
        time.sleep(1)
        moveServo(1, 1000, 500)
        time.sleep(1)
        # runActionGroup(0,1000)
