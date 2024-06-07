# 第二步数据集预处理，转为numpy格式存储，然后再去运行train.py
import numpy as np
from PIL import Image
# import pdb
import os
# 数据集预处理
data_dir = 'F:\\my_dataset\\NWPU-ReID\\'
data_path = 'F:\\my_dataset\\NWPU-ReID\\ori_data'  # 未处理的数据集路径

rgb_cameras = ['cam2', 'cam3', 'cam6', 'cam7']  # 这些是RGB摄像机
ir_cameras = ['cam1', 'cam4', 'cam5', 'cam8']  # 这些是红外摄像机

# load id info
file_path_train = os.path.join(data_path, 'exp/train_id.txt')  # 这个文本里放着作为训练集的行人ID
with open(file_path_train, 'r') as file:  # 打开训练集ID的txt
    ids = file.read().splitlines()  # 按行分割 但其实就一行 得到的是含有1个元素的列表
    ids = [int(y) for y in ids[0].split(',')]  # 这把ID做成了列表形式 [1, 2, 4, 5, ..., 532, 533] 类似这样的 元素是int类型
    id_train = ["%04d" % x for x in ids]  # 把ID扩充为4位 [‘0001’, ‘0002, ’0004‘, ’0005‘, ..., ’0532’, ‘0533’] 元素是字符串类型


# 经过前两个with，print的话可以看到， 0001-0333和0434-0533为训练集，编号0334-0433为验证集 俩集合没有交集

files_rgb = []
files_ir = []
for id in sorted(id_train):  # 注意，sorted(id_train) 把id_train的ID从小到大排列了，此时验证集ID，夹在训练集ID，中间
    for cam in rgb_cameras:  # 'cam1', 'cam2', 'cam4', 'cam5'
        img_dir = os.path.join(data_path, cam, id)  # 形如: SYSU-MM01/ori_data\cam1\0001
        if os.path.isdir(img_dir):  # 这里判断那个人的文件夹是否存在，因为实际上，txt文本里的id远远超过数据集的id数量
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])  # 这个人，比如0001，的所有图片的路径，做成列表并按顺序排好
            files_rgb.extend(new_files)  # 把前一行的那个列表拼到files_rgb里面 所有cam的所有行人rgb图片
            
    for cam in ir_cameras:  # 红外的也是一样操作
        img_dir = os.path.join(data_path, cam, id)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)

# relabel
pid_container = set()  # 创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
for img_path in files_ir:
    pid = int(img_path[-43:-39])  # 获取行人编号，形如: 1 ，int类型的
    pid_container.add(pid)  # {1, 2,...}
pid2label = {pid: label for label, pid in enumerate(pid_container)}  # 形如{1: 0, 2: 1, 3: 2, 4: 3, 5: 4} label是下标
fix_image_width = 144  # 图片宽度
fix_image_height = 288  # 图片高度 高：宽=2：1


def read_imgs(train_image):
    train_img = []
    train_label = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)  # 打开图片
        img = img.convert('RGB')  # 如果图片为PNG还需要这个步骤
        img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)  # 修改图片尺寸，Image.ANTIALIAS 代表高质量
        pix_array = np.array(img)  # 转为numpy文件

        train_img.append(pix_array)  # 所有img的numpy拼接

        # label
        pid = int(img_path[-43:-39])
        pid = pid2label[pid]  # 获得编号 在label.npy文件里的下标 会有重复的 因为行人在不同摄像头下
        train_label.append(pid)  # 编号拼接
    return np.array(train_img), np.array(train_label)  # 都转为numpy


# rgb images
train_img, train_label = read_imgs(files_rgb)  # files_rgb是所有RGB文件
print('正在保存rgb文件')
np.save(data_dir + 'train_rgb_resized_img.npy', train_img)
np.save(data_dir + 'train_rgb_resized_label.npy', train_label)

# ir images
train_img, train_label = read_imgs(files_ir)
print('正在保存ir文件')
np.save(data_dir + 'train_ir_resized_img.npy', train_img)
np.save(data_dir + 'train_ir_resized_label.npy', train_label)
