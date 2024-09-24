import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import re

import serial


class log:
    def __init__(self):
        self.LOG_PATH = "../Things/Logs/"
        self.ser = serial.Serial("COM12", 921600)
        self.RE_PATTERN = r"T0(\w+)/\[(.*?)\](.*?)#"
        self.receive_thread = threading.Thread(target=self.receive_data, daemon=True)
        self.frame_count = 0

    def receive_data(self):
        _frame = bytearray()
        while True:
            _temp = self.ser.read()
            _frame += _temp
            if _temp == b"#" and self.ser.read() == b"T":
                self.update_text(_frame)
                _frame.clear()
                _frame.append(0x54)

        # while True:
        #     # 模拟接收新数据
        #     print("receive new data")
        #     data = bytearray(b"T0INFO/[0.100s] armTaskStart is running-")
        #     self.root.after(0, self.update_text, data)
        #     time.sleep(2)

        #     data = bytearray(b"T0WARNING/[0.100s] armTaskStart is running-")
        #     self.root.after(0, self.update_text, data)
        #     time.sleep(2)

        #     data = bytearray(b"T0ERROR/[0.100s] armTaskStart is running-")
        #     self.root.after(0, self.update_text, data)
        #     time.sleep(2)

    def update_text(self, _data):
        temp_frame = {
            "type": None,
            "time": None,
            "content": None,
        }
        _text = _data.decode("utf-8")
        _match = re.search(self.RE_PATTERN, _text)
        if _match:
            temp_frame["type"] = _match.group(1)
            temp_frame["time"] = _match.group(2)
            temp_frame["content"] = _match.group(3)

        self.text_area.insert(
            tk.END,
            "Times: " + str(temp_frame["time"]) + "   " + str(self.frame_count) + "\n",
        )
        print(22222)
        # self.text_area.tag_config("a", foreground="green")
        self.frame_count += 1

        if temp_frame["type"] == "INFO":

            font_type = "blue"
        elif temp_frame["type"] == "WARNING":
            font_type = "orange"
        elif temp_frame["type"] == "ERROR":
            font_type = "red"
        else:
            font_type = "black"

        self.text_area.insert(
            tk.END,
            "[" + str(temp_frame["type"]) + "]: " + str(temp_frame["content"]) + "\n",
            font_type,
        )
        self.text_area.tag_config(font_type, foreground=font_type)

        self.text_area.see(tk.END)
        # E:\project\User\Things
        with open(
            r"E:\project\User\Things\Logs\Log_" + self.file_name + ".txt", "a"
        ) as file:
            file.write(
                "Times: "
                + str(temp_frame["time"])
                + "   "
                + str(self.frame_count)
                + "\n"
            )
            file.write(
                "["
                + str(temp_frame["type"])
                + "]: "
                + str(temp_frame["content"])
                + "\n\n"
            )

    def UI_init(self):
        # 初始化主窗口
        self.root = tk.Tk()
        self.root.title("Serial Port and Data Display")

        # 创建一个可滚动的文本框，并设置字体大小
        font_settings = ("Helvetica", 20)
        self.text_area = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, width=60, height=20, font=font_settings
        )
        self.text_area.pack(pady=10, padx=10)

        # 创建一个Frame用于排列下拉菜单
        self.selection_frame = tk.Frame(self.root)
        self.selection_frame.pack(pady=10)

    def start(self):
        self.UI_init()
        name = time.localtime()
        self.file_name = (
            str(name.tm_mon)
            + "_"
            + str(name.tm_mday)
            + " "
            + str(name.tm_hour)
            + "h"
            + str(name.tm_min)
            + "m"
        )
        print("Log file name: " + self.file_name)
        print("Mainloop exited")
        self.receive_thread.start()
        self.root.mainloop()


def receive_data():
    while True:
        # 模拟接收新数据
        data = bytearray(b"T0INFO/[0.100s] armTaskStart is running-")
        # 在主线程中更新UI
        root.after(0, update_text, data)
        # 模拟接收间隔
        time.sleep(2)


def select_port():
    selected_port = port_var.get()
    print(f"Selected Port: {selected_port}")


def select_baudrate():
    selected_baudrate = baudrate_var.get()
    print(f"Selected Baudrate: {selected_baudrate}")


if __name__ == "__main__":
    log_obj = log()
    log_obj.start()
