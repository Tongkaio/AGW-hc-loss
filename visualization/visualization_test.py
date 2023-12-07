import numpy as np
# import matplotlib.pyplot as plt
# fig=plt.figure(figsize=(4,3), facecolor='blue')
# plt.show()
from PIL import Image

pic_path = r'D:\pythoncode\NWPU-ReID\ori_data\cam1\0001\0001_2022_03_13_09_51_34_1_0_0_010.png'
img = Image.open(pic_path)  # test_img_file[i]将获得具体图片的地址
img = img.convert('RGB')  # 如果图片为PNG还需要这个步骤
print(img)
img.show()