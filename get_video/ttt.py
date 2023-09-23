import numpy as np
import cv2 as cv
img=cv.imread(r"C:\Users\WYM\Web\CaptureFiles\2021-11-25\20211126084735354.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# ss=cv.imread(r"D:\dataset\RegDB\Thermal\2\female_back_t_06300_2.bmp")
iii=np.array(img)
# iii2=np.array(ss)
print(iii.shape)
print(iii)
# print(iii2.shape)