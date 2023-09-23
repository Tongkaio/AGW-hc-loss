# -*- coding: utf-8 -*-
'''
播放库调用文件
'''
__author__ = 'Dean'

from ctypes import *
import ctypes
import os

# 显示回调函数
DISPLAYCBFUN = WINFUNCTYPE(None, c_long, c_char_p, c_long, c_long, c_long, c_long, c_long, c_long)
# 显示回调函数linux
#DISPLAYCBFUN = CFUNCTYPE(None, c_long, c_char_p, c_long, c_long, c_long, c_long, c_long, c_long)

PLAYCTRL_PORT = c_long(-1)
Playctrldll = None
FuncDisplayCB = None

def LoadPlayctrlSDK(sdkPath, windowsFlag):
    '''
    加载PlayctrlSDK库
    '''
    global Playctrldll

    if not windowsFlag:
        Playctrldll = cdll.LoadLibrary(sdkPath + r'/libPlayCtrl.so')
    else:
        # Playctrldll = libc

        Playctrldll = WinDLL(r"./PlayCtrl.dll")
        # Playctrldll = WinDLL(r'D:\app_installation\SDK\python\PythonDemo\lib\PlayCtrl.dll')
    return Playctrldll

def Playctrl_Getport(Playctrldll):
    '''
    获取未使用的通道号
    '''
    Playctrldll.PlayM4_GetPort(byref(PLAYCTRL_PORT))

    if PLAYCTRL_PORT.value < 0:
        return False
    return True

def DisplayCBFun(nPort, pBuf, nSize, nWidth, nHeight, nStamp, nType, nReserved):
    '''
    显示回调函数
    '''
    # 2021/12/5 我注释掉了这个语句，并添加了一行内容:pass
    pass
    # print(nWidth, nHeight)


