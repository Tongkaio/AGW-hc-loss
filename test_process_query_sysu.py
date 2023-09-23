import os
ori_data_path = 'NWPU-ReID/ori_data'
for cam in os.listdir(ori_data_path):
    cam_path = os.path.join(ori_data_path, cam)
    pic_num = len(os.listdir(cam_path))
    print('{} has {} pics'.format(cam, pic_num))