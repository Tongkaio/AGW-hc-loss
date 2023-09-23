# 第一步划分数据集
import random
Person_num = 600  # 这里输入总人数
resultList = []  # 用于存放结果的List
A = 1
B = Person_num
COUNT = Person_num
resultList = random.sample(range(A, B+1), COUNT)  # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素
train_list = sorted(resultList[0:480])
# val_list = sorted(resultList[187:247])
test_list = sorted(resultList[480:Person_num])

train_str = ",".join(str(i) for i in train_list)
# val_str = ",".join(str(i) for i in val_list)
test_str = ",".join(str(i) for i in test_list)
with open(r'F:\my_dataset\NWPU-ReID\ori_data\exp\train_id.txt', 'w') as wr:
    wr.write(train_str)
# with open(r'F:\my_dataset\NWPU-ReID\ori-data\exp\val_id.txt', 'w') as wr:
#     wr.write(val_str)
with open(r'F:\my_dataset\NWPU-ReID\ori_data\exp\test_id.txt', 'w') as wr:
    wr.write(test_str)
print('训练集{}人,ID:{}'.format(len(train_list), train_str))
# print('验证集{}人,ID:{}'.format(len(val_list), val_str))
print('测试集{}人,ID:{}'.format(len(test_list), test_str))
