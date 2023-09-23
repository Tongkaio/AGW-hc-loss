# -*- coding: utf-8 -*-
'''
HCNetSDK头文件,定义机构体
'''

__author__ = 'Dean'

from ctypes import *

# 定义登录结构体
class NET_DVR_DEVICEINFO_V30(Structure):
    pass
LPNET_DVR_DEVICEINFO_V30 = POINTER(NET_DVR_DEVICEINFO_V30)
NET_DVR_DEVICEINFO_V30._fields_ = [
    ('sSerialNumber', c_ubyte * 48),
    ('byAlarmInPortNum', c_ubyte),
    ('byAlarmOutPortNum', c_ubyte),
    ('byDiskNum', c_ubyte),
    ('byDVRType', c_ubyte),
    ('byChanNum', c_ubyte),
    ('byStartChan', c_ubyte),
    ('byAudioChanNum', c_ubyte),
    ('byIPChanNum', c_ubyte),
    ('byZeroChanNum', c_ubyte),
    ('byMainProto', c_ubyte),
    ('bySubProto', c_ubyte),
    ('bySupport', c_ubyte),
    ('bySupport1', c_ubyte),
    ('bySupport2', c_ubyte),
    ('wDevType', c_ushort),
    ('bySupport3', c_ubyte),
    ('byMultiStreamProto', c_ubyte),
    ('byStartDChan', c_ubyte),
    ('byStartDTalkChan', c_ubyte),
    ('byHighDChanNum', c_ubyte),
    ('bySupport4', c_ubyte),
    ('byLanguageType', c_ubyte),
    ('byVoiceInChanNum', c_ubyte),
    ('byStartVoiceInChanNo', c_ubyte),
    ('bySupport5', c_ubyte),
    ('bySupport6', c_ubyte),
    ('byMirrorChanNum', c_ubyte),
    ('wStartMirrorChanNo', c_ushort),
    ('bySupport7', c_ubyte),
    ('byRes2', c_ubyte),
]

# 定义组件库加载路径信息结构体
class NET_DVR_LOCAL_SDK_PATH(Structure):
    pass
LPNET_DVR_LOCAL_SDK_PATH = POINTER(NET_DVR_LOCAL_SDK_PATH)
NET_DVR_LOCAL_SDK_PATH._fields_ = [
    ('sPath', c_char * 256),
    ('byRes', c_ubyte * 128),
]

# 定义预览参数结构体
class NET_DVR_PREVIEWINFO(Structure):
    pass
LPNET_DVR_PREVIEWINFO = POINTER(NET_DVR_PREVIEWINFO)
NET_DVR_PREVIEWINFO._fields_ = [
    ('lChannel', c_long),
    ('dwStreamType', c_ulong),
    ('dwLinkMode', c_ulong),
    ('hPlayWnd', c_void_p),
    ('bBlocked', c_ulong),
    ('bPassbackRecord', c_ulong),
    ('byPreviewMode', c_ubyte),
    ('byStreamID', c_ubyte * 32),
    ('byProtoType', c_ubyte),
    ('byRes1', c_ubyte),
    ('byVideoCodingType', c_ubyte),
    ('dwDisplayBufNum', c_ulong),
    ('byRes', c_ubyte * 216),
]

# 码流回调数据类型
NET_DVR_SYSHEAD = 1
NET_DVR_STREAMDATA = 2
NET_DVR_AUDIOSTREAMDATA = 3
NET_DVR_PRIVATE_DATA = 112

# 码流回调函数
REALDATACALLBACK = WINFUNCTYPE(None, c_long, c_ulong, POINTER(c_ubyte), c_ulong, c_void_p)
# 码流回调函数 Linux
#REALDATACALLBACK = CFUNCTYPE(None, c_long, c_ulong, POINTER(c_ubyte), c_ulong, c_void_p)

# 云台控制命令
LIGHT_PWRON = 2  #接通灯光电源
WIPER_PWRON = 3  #接通雨刷开关
FAN_PWRON = 4  #接通风扇开关
HEATER_PWRON = 5  #接通加热器开关
AUX_PWRON1 = 6  #接通辅助设备开关
AUX_PWRON2 = 7  #接通辅助设备开关
ZOOM_IN = 11  #焦距变大(倍率变大)
ZOOM_OUT = 12  #焦距变小(倍率变小)
FOCUS_NEAR = 13  #焦点前调
FOCUS_FAR = 14  #焦点后调
IRIS_OPEN = 15  #光圈扩大
IRIS_CLOSE = 16  #光圈缩小
TILT_UP = 21  #云台上仰
TILT_DOWN = 22  #云台下俯
PAN_LEFT = 23  #云台左转
PAN_RIGHT = 24  #云台右转
UP_LEFT = 25  #云台上仰和左转
UP_RIGHT = 26  #云台上仰和右转
DOWN_LEFT = 27  #云台下俯和左转
DOWN_RIGHT = 28  #云台下俯和右转
PAN_AUTO = 29  #云台左右自动扫描
TILT_DOWN_ZOOM_IN  = 58  #云台下俯和焦距变大(倍率变大)
TILT_DOWN_ZOOM_OUT = 59  #云台下俯和焦距变小(倍率变小)
PAN_LEFT_ZOOM_IN = 60  #云台左转和焦距变大(倍率变大)
PAN_LEFT_ZOOM_OUT = 61  #云台左转和焦距变小(倍率变小)
PAN_RIGHT_ZOOM_IN = 62  #云台右转和焦距变大(倍率变大)
PAN_RIGHT_ZOOM_OUT = 63  #云台右转和焦距变小(倍率变小)
UP_LEFT_ZOOM_IN = 64  #云台上仰和左转和焦距变大(倍率变大)
UP_LEFT_ZOOM_OUT = 65  #云台上仰和左转和焦距变小(倍率变小)
UP_RIGHT_ZOOM_IN = 66  #云台上仰和右转和焦距变大(倍率变大)
UP_RIGHT_ZOOM_OUT = 67  #云台上仰和右转和焦距变小(倍率变小)
DOWN_LEFT_ZOOM_IN = 68  #云台下俯和左转和焦距变大(倍率变大)
DOWN_LEFT_ZOOM_OUT = 69  #云台下俯和左转和焦距变小(倍率变小)
DOWN_RIGHT_ZOOM_IN  = 70  #云台下俯和右转和焦距变大(倍率变大)
DOWN_RIGHT_ZOOM_OUT = 71  #云台下俯和右转和焦距变小(倍率变小)
TILT_UP_ZOOM_IN = 72  #云台上仰和焦距变大(倍率变大)
TILT_UP_ZOOM_OUT = 73  #云台上仰和焦距变小(倍率变小)
