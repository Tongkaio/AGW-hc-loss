from __future__ import print_function, absolute_import
import os
import numpy as np
import random


def process_query_sysu(data_path, mode='all', relabel=False, v2t=2):  # 'SYSU-MM01/ori_data/'
    if v2t == 2:
        if mode == 'all':
            cameras = ['cam3', 'cam6']  # 都是红外相机
        elif mode == 'indoor':
            cameras = ['cam3', 'cam6']  # 都是红外相机
    if v2t == 1:
        if mode == 'all':
            if mode == 'all':
                cameras = ['cam1', 'cam2', 'cam4', 'cam5']
            elif mode == 'indoor':
                cameras = ['cam1', 'cam2']
    
    file_path = os.path.join(data_path, 'exp/test_id.txt')  # 这是在训练阶段用于验证的 'SYSU-MM01/ori_data/exp/test_id.txt'
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])  # SYSU-MM01/ori_data/cam1/0001/0001.jpg
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)


def process_query_nwpu(data_path, mode='all', relabel=False, val=0, v2t=2):  # 'SYSU-MM01/ori_data/'
    if v2t == 2:
        if mode == 'all':  #
            cameras = ['cam1', 'cam4', 'cam5', 'cam8']  # 全部红外相机
        elif mode == 'indoor':
            cameras = ['cam1', 'cam4']  # 室内的红外相机
    if v2t == 1:
        if mode == 'all':  #
            cameras = ['cam2', 'cam3', 'cam6', 'cam7']  # 全部RGB相机
        elif mode == 'indoor':
            cameras = ['cam2', 'cam3']  # 室内的RGB相机
    if val == 1:
        file_path = os.path.join(data_path, 'exp/val_id.txt')  # 这是在训练阶段用于验证的 'SYSU-MM01/ori_data/exp/test_id.txt'
    if val == 0:
        file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_rgb = []
    files_ir = []

    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_ir.extend(new_files)
    query_img = []
    query_id = []
    query_cam = []
    for img_path in files_ir:
        camid, pid = int(img_path[-45]), int(img_path[-43:-39])  # SYSU-MM01/ori_data/cam1/0001/0001.jpg
        query_img.append(img_path)
        query_id.append(pid)
        query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)

def process_gallery_sysu(data_path, mode = 'all', trial = 0, relabel=False, v2t=1):
    random.seed(trial)
    if v2t == 2:
        if mode == 'all':
            cameras = ['cam3', 'cam6']  # 都是红外相机
        elif mode == 'indoor':
            cameras = ['cam3', 'cam6']  # 都是红外相机
    if v2t == 1:
        if mode == 'all':
            if mode == 'all':
                cameras = ['cam1', 'cam2', 'cam4', 'cam5']
            elif mode == 'indoor':
                cameras = ['cam1', 'cam2']
        
    file_path = os.path.join(data_path, 'exp/test_id.txt')
    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path,cam,id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path[-15]), int(img_path[-13:-9])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)


def process_gallery_nwpu(data_path, mode='all', trial=0, relabel=False, val=0, v2t=1):
    random.seed(trial)
    if v2t == 2:
        if mode == 'all':  #
            cameras = ['cam1', 'cam4', 'cam5', 'cam8']  # 全部红外相机
        elif mode == 'indoor':
            cameras = ['cam1', 'cam4']  # 室内的红外相机
    if v2t == 1:
        if mode == 'all':  #
            cameras = ['cam2', 'cam3', 'cam6', 'cam7']  # 全部RGB相机
        elif mode == 'indoor':
            cameras = ['cam2', 'cam3']  # 室内的RGB相机

    if val == 1:
        file_path = os.path.join(data_path, 'exp/val_id.txt')  # 这是在训练阶段用于验证的 'SYSU-MM01/ori_data/exp/test_id.txt'
    if val == 0:
        file_path = os.path.join(data_path, 'exp/test_id.txt')

    files_rgb = []
    with open(file_path, 'r') as file:
        ids = file.read().splitlines()
        ids = [int(y) for y in ids[0].split(',')]
        ids = ["%04d" % x for x in ids]

    for id in sorted(ids):
        for cam in cameras:
            img_dir = os.path.join(data_path, cam, id)
            if os.path.isdir(img_dir):
                new_files = sorted([img_dir + '/' + i for i in os.listdir(img_dir)])
                files_rgb.append(random.choice(new_files))  # 随即返回一张，从摄像机的for循环可以看出是对每个摄像机操作一次
    gall_img = []
    gall_id = []
    gall_cam = []
    for img_path in files_rgb:
        camid, pid = int(img_path[-45]), int(img_path[-43:-39])
        gall_img.append(img_path)
        gall_id.append(pid)
        gall_cam.append(camid)
    return gall_img, np.array(gall_id), np.array(gall_cam)


def process_test_regdb(img_dir, trial=1, modal='visible'):
    if modal == 'visible':
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal == 'thermal':
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'
    
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, np.array(file_label)