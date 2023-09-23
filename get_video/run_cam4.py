# -*- coding: utf-8 -*-
"""
主函数入口文件
wym: python 3
"""
__author__ = 'Dean'

# 摄像头4 室内 红外双摄
import platform
import threading
import time
import tkinter  # GUI工具包

from HCNetSDK_header import *
from playctrl import *
import datetime
# import os # 不需要了，在platform里已经import os了

# 这些根据你自己的路由以及账号的设置进行修改
# 设备信息申明
# create_string_buffer是ctype的函数，用来创建字符串，使得字符串可被修改，来给C++的函数使用。
CAM_ID = '4'  # 摄像头编号
ENVIRONMENT = '_indoor'  # 环境
DEV_IP = create_string_buffer(b'192.168.1.65')  # 网络摄像头的IP地址
DEV_PORT = 8000  # 网络摄像头的端口号
DEV_USER_NAME = create_string_buffer(b'admin')  # 网络摄像头的登录名
DEV_PASSWORD = create_string_buffer(b'flyfordream208')  # 网络摄像头的登录密码
WINDOWS_FLAG = True  # 默认系统为Windows系统

win = None
funcRealDataCallBack_V30 = None
lRealPlayHandle = None
video_count = None  # 记录录像个数
# HCNetSDK库文件路径
HCNETSDK_DLL_PATH = r'./HCNetSDK.dll'
HCNETSDK_DLL_PATH_LINUX = r'./libhcnetsdk.so'


def GetPlatform():
    sysstr = platform.system()  # 获取操作系统类型，返回值是Windows、Linus等
    if sysstr != "Windows":  # 如果不是Windows系统，把此FLAG置False
        global WINDOWS_FLAG  # 调用全局变量WINDOWS_FLAG
        WINDOWS_FLAG = False  # 置False


def LoadHCNetSDK():
    """
    加载HCNetSDK库
    """
    if not WINDOWS_FLAG:
        Objdll = cdll.LoadLibrary(HCNETSDK_DLL_PATH_LINUX)
    else:
        Objdll = WinDLL(HCNETSDK_DLL_PATH)
    return Objdll


def InitHCNetSDK(Objdll):
    """
    初始化HCNetSDK库
    """
    if not WINDOWS_FLAG:
        sdk_path = NET_DVR_LOCAL_SDK_PATH()
        sdk_path.sPath = os.path.dirname(HCNETSDK_DLL_PATH).encode('utf-8')
        Objdll.NET_DVR_SetSDKInitCfg(2, byref(sdk_path))

    # 初始化DLL
    Objdll.NET_DVR_Init()  # 这一行才是[真]·初始化
    # 设置日志路径
    if not WINDOWS_FLAG:
        Objdll.NET_DVR_SetLogToFile(3, create_string_buffer(b'/home/sdklog'), True)
    else:
        Objdll.NET_DVR_SetLogToFile(3, create_string_buffer(b'G:\\SdkLog'), True)
    # 设置设备超时时间
    # Objdll.NET_DVR_SetConnectTime(int(1000), 1)


def LoginDev(Objdll):
    """
    登录设备
    """
    device_info = NET_DVR_DEVICEINFO_V30()  # 设备信息
    lUserId = Objdll.NET_DVR_Login_V30(DEV_IP, DEV_PORT, DEV_USER_NAME, DEV_PASSWORD, byref(device_info))
    return (lUserId, device_info)


def RealDataCallBack_V30(lPlayHandle, dwDataType, pBuffer, dwBufSize, pUser):
    """
    码流回调函数
    """
    if dwDataType == NET_DVR_SYSHEAD:
        # 设置流播放模式
        Playctrldll.PlayM4_SetStreamOpenMode(PLAYCTRL_PORT, 0)
        if Playctrldll.PlayM4_OpenStream(PLAYCTRL_PORT, pBuffer, dwBufSize, 1024 * 1000):
            global FuncDisplayCB
            FuncDisplayCB = DISPLAYCBFUN(DisplayCBFun)
            Playctrldll.PlayM4_SetDisplayCallBack(PLAYCTRL_PORT, FuncDisplayCB)
            if Playctrldll.PlayM4_Play(PLAYCTRL_PORT, cv.winfo_id()):
                pass
                # print(u'播放库播放成功')
                # print('------------------------')
            else:
                print(u'播放库播放失败')
                print('------------------------')
        else:
            print(u'播放库打开流失败')
            print('------------------------')
    elif dwDataType == NET_DVR_STREAMDATA:
        Playctrldll.PlayM4_InputData(PLAYCTRL_PORT, pBuffer, dwBufSize)

    else:
        print(u'其他数据,长度:', dwBufSize)


def OpenPreview(Objdll, lUserId, callbackFun):
    """
    打开预览
    """
    preview_info = NET_DVR_PREVIEWINFO()
    preview_info.hPlayWnd = None
    preview_info.lChannel = 1  # 通道号
    preview_info.dwStreamType = 0  # 主码流
    preview_info.dwLinkMode = 0  # TCP
    preview_info.bBlocked = 1  # 阻塞取流

    lRealPlayHandle = Objdll.NET_DVR_RealPlay_V40(lUserId, byref(preview_info), callbackFun, None)

    return lRealPlayHandle


def CaptureVideo():
    """
    开始录像
    """
    global video_count
    video_count = 0
    # 录像保存地址和命名
    # print('---------------------------------------------------------')
    # print("摄像机编号:%s" % CAM_ID)
    while True:
        video_count = video_count + 1  # 计数加1
        videoname_root = r"J:\video_capture\cam4"  # 录像保存的文件夹
        # datetime.datetime.now()返回本地时间格式: 2021-12-05 15.53.28.852835
        time_now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
        if int(time_now.split('_')[3]) > 18 or int(time_now.split('_')[3]) < 7:  # 晚上7点至次日6点为夜晚 其余白天
            day_or_night = 'night'  # 晚上
        else:
            day_or_night = 'day'  # 白天
        v_path = time_now + "_cam" + CAM_ID + "_" + day_or_night + ENVIRONMENT + ".mp4"  # 文件名
        path = os.path.join(videoname_root, v_path)  # 文件路径指针，包括文件名，这一步是把文件夹路径和文件名拼接起来

        if Objdll.NET_DVR_SaveRealData(lRealPlayHandle, bytes(path, 'utf-8')):  # 开启一次录像
            start_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("cam%s第%d段录像已开始 %s" % (CAM_ID, video_count, start_time))
            win.title('监控预览  cam%s-第%d段录像' % (CAM_ID, video_count))
        else:
            print("cam%s录像失败" % CAM_ID)
        time.sleep(300)  # 间隔时间，即每多少秒保存一次录像

        Objdll.NET_DVR_StopSaveRealData(lRealPlayHandle)  # 结束一次录像
        end_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('cam%s第%d段录像已保存 %s %s' % (CAM_ID, video_count, end_time, v_path))


def InputData(fileMp4, Playctrldll):
    while True:
        pFileData = fileMp4.read(4096)
        if pFileData is None:
            break

        if not Playctrldll.PlayM4_InputData(PLAYCTRL_PORT, pFileData, len(pFileData)):
            break


def real_play():
    global lRealPlayHandle
    lRealPlayHandle = OpenPreview(Objdll, lUserId, funcRealDataCallBack_V30)
    if lRealPlayHandle < 0:  # 预览错误
        print('Open preview fail, error code is:', Objdll.NET_DVR_GetLastError())
        # 登出设备
        Objdll.NET_DVR_Logout(lUserId)
        # 释放资源
        Objdll.NET_DVR_Cleanup()
        print('预览失败，程序意外地结束')
        exit()


if __name__ == '__main__':
    win = tkinter.Tk()  # 创建窗口
    win.title('监控预览cam%s' % CAM_ID)
    win.resizable(0, 0)  # 固定窗口大小，resizable（宽，高），宽高都为0，代表宽高都不能改变大小，即固定大小。
    win.overrideredirect(0)  # 括号里若填0则窗口边框出现，若填1则省略边框
    sw = win.winfo_screenwidth()  # 得到屏幕(物理)宽度 1536
    sh = win.winfo_screenheight()  # 得到屏幕(物理)高度 864
    ww = 512  # 设定窗口宽度
    wh = 384  # 设定窗口高度
    # 窗口宽高为100
    x = (sw - ww) / 2  # (1536-512)/2=512
    y = (sh - wh) / 2  # (864-384)/2=240
    win.geometry("%dx%d+%d+%d" % (ww, wh, x, y))  # 四个参数为：宽、高、左、上，其中×号连接的两个数代表窗口大小，而+号则是移动串口位置

    # 创建一个Canvas，设置其背景色为白色
    cv = tkinter.Canvas(win, bg='white', width=ww, height=wh)
    cv.pack()

    # 获取系统平台
    GetPlatform()  # WINDOWS_FLAG 为 True
    # 导入库
    (sdkPath, tempfilename) = os.path.split(HCNETSDK_DLL_PATH)  # 分离出路径、文件名
    Playctrldll = LoadPlayctrlSDK(sdkPath, WINDOWS_FLAG)  # 导入播放库

    if not Playctrl_Getport(Playctrldll):  # 获取播放库通道号
        print(u'获取播放库通道号失败')
        exit()

    ##############################################################################################

    # 加载HCNetSDK库
    Objdll = LoadHCNetSDK()

    # 初始化HCNetSDK库
    InitHCNetSDK(Objdll)

    # 登录设备
    (lUserId, device_info) = LoginDev(Objdll)  # 返回设备登陆信息（IP，端口，用户名，密码）和设备信息（device_info）
    if lUserId < 0:  # 错误!
        time.sleep(5)
        (lUserId, device_info) = LoginDev(Objdll)
        if lUserId < 0:
            print('Login device fail, error code is:', Objdll.NET_DVR_GetLastError())
            # 释放资源
            Objdll.NET_DVR_Cleanup()
            print('cam%s登陆失败，程序意外地结束' % CAM_ID)
            exit()  # 结束程序

    # 定义码流回调函数
    funcRealDataCallBack_V30 = REALDATACALLBACK(RealDataCallBack_V30)

    ##############################################################################################
    # 开启预览线程
    win_thread = threading.Thread(target=real_play, name='real_play')
    win_thread.setDaemon(True)  # 主线程结束后杀死此线程
    win_thread.start()  # 线程启动
    win_thread.join()

    # 开启录像线程
    video_thread = threading.Thread(target=CaptureVideo, name='CaptureVideo')
    video_thread.setDaemon(True)  # 主线程结束后杀死此线程
    video_thread.start()  # 线程启动

    win.mainloop()  # 让预览窗口循环显示

    # 以下函数，用于停止所有工作
    Playctrldll.PlayM4_Stop(PLAYCTRL_PORT)
    Playctrldll.PlayM4_CloseStream(PLAYCTRL_PORT)
    Playctrldll.PlayM4_FreePort(PLAYCTRL_PORT)
    PLAYCTRL_PORT = c_long(-1)
    # 关闭录像
    Objdll.NET_DVR_StopSaveRealData(lRealPlayHandle)
    print('cam%s停止录像' % CAM_ID)
    # 关闭预览
    Objdll.NET_DVR_StopRealPlay(lRealPlayHandle)
    # 登出设备
    Objdll.NET_DVR_Logout(lUserId)
    # 释放资源
    Objdll.NET_DVR_Cleanup()
    print('cam%s程序正常地结束' % CAM_ID)
