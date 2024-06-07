import os
import threading
import time
# 运行此脚本以调用全部摄像头


def thread_cam1():
    os.system("python run_cam1.py")

def thread_cam2():
    os.system("python run_cam2.py")

def thread_cam3():
    os.system("python run_cam3.py")

def thread_cam4():
    os.system("python run_cam4.py")

def thread_cam5():
    os.system("python run_cam5.py")

def thread_cam6():
    os.system("python run_cam6.py")

def thread_cam7():
    os.system("python run_cam7.py")

def thread_cam8():
    os.system("python run_cam8.py")

def main():
    added_thread = threading.Thread(target=thread_cam1)
    added_thread.start()
    time.sleep(3)
    added_thread = threading.Thread(target=thread_cam2)
    added_thread.start()
    time.sleep(3)
    added_thread = threading.Thread(target=thread_cam3)
    added_thread.start()
    time.sleep(3)
    added_thread = threading.Thread(target=thread_cam4)
    added_thread.start()
    time.sleep(3)
    added_thread = threading.Thread(target=thread_cam5)
    added_thread.start()
    time.sleep(3)
    added_thread = threading.Thread(target=thread_cam6)
    added_thread.start()
    time.sleep(3)
    added_thread = threading.Thread(target=thread_cam7)
    added_thread.start()
    time.sleep(3)
    added_thread = threading.Thread(target=thread_cam8)
    added_thread.start()
    time.sleep(3)


if __name__ == '__main__':
    main()
