import threading
import time

num = 0


def run():
    global num
    for i in range(10):
        time.sleep(0.5)
        num += 1
        print(f"Thread {threading.current_thread().name} has incremented the number to {num}")
        if i == 6:
            print(f"Thread {threading.current_thread().name} is about to wait the event")
            event.wait()


if __name__ == "__main__":
    event = threading.Event()
    t1 = threading.Thread(target=run, name="Thread 1")
    t2 = threading.Thread(target=run, name="Thread 2")
    t1.start()
    t2.start()

    time.sleep(8)
    print("event is set")

    event.clear()
    print("event is cleared")


