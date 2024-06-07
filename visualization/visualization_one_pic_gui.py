from __future__ import print_function
import argparse
import time
import cv2
import matplotlib
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_nwpu
from model import embed_net
from utils import *
import matplotlib.pyplot as plt
import pdb

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='nwpu', help='dataset name: regdb or sysu or nwpu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline: resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='awg', type=str,
                    metavar='m', help='method type: base or awg')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')
parser.add_argument('--tvsearch', action='store_true', help='whether thermal to visible search on RegDB')

def process_one_query_nwpu(pic_path):
    img_path = pic_path
    query_img = []
    query_id = []
    query_cam = []
    camid, pid = int(img_path[-45]), int(img_path[-43:-39])
    query_img.append(img_path)
    query_id.append(pid)
    query_cam.append(camid)
    return query_img, np.array(query_id), np.array(query_cam)


def extract_gall_feat(gall_loader, net, ngall, pool_dim, test_mode):
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool, feat_fc = net(input, input, test_mode[0])
            gall_feat_pool[ptr:ptr + batch_num, :] = feat_pool.detach().cpu().numpy()
            gall_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat_pool, gall_feat_fc


def extract_query_feat(query_loader, net, nquery, pool_dim, test_mode):
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):  # 返回值batch序号（0-63）， 一个batch_num的图像，一个batch_num的图像ID
            batch_num = input.size(0)  # 有多少张 batchsize * channel * width * height   0指的是batchsize
            input = Variable(input.cuda())
            feat_pool, feat_fc = net(input, input, test_mode[1])
            query_feat_pool[ptr:ptr + batch_num,
            :] = feat_pool.detach().cpu().numpy()  # 网络里的值都是有梯度的值 为了转化为numpy需要去掉梯度 常用的方法就是 detach().cpu().numpy() https://blog.csdn.net/xiongzai2016/article/details/107175435
            query_feat_fc[ptr:ptr + batch_num, :] = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat_pool, query_feat_fc


def visualization(distmat, q_pids, g_pids, q_camids, g_camids, g_imgs, q_img):  # 相似度
    # -distmat_att, query_label, gall_label, query_cam, gall_cam
    num_q, num_g = distmat.shape  # 相似度矩阵的行列分别代表query数量和gallery数量  num_q=3851  num_g=240
    indices = np.argsort(distmat, axis=1)  # 将x中的元素从小到大排列，提取其对应的index(索引)，即相似度排名
    pred_label = g_pids[indices]  # 预测的标签，这里将返回
    # print(q_pids)
    i = 1
    target_gallery_path = []
    labeltext = []
    similartiytext = []
    for pos, label, similartiy in zip(indices[0][0:10], pred_label[0][0:10], -distmat[0][indices][0]):
        target_gallery_path.append(g_imgs[pos])
        # labelwithsimilar.append(str(label) + '/' + format(similartiy*100,'.1f') + '%')
        labeltext.append(str(label))
        similartiytext.append(format(similartiy*100,'.1f') + '%')
    return target_gallery_path, labeltext, similartiytext


def predict(pic_path, resume, mode='t2v', data_path='../NWPU-ReID/ori_data/'):
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.resume = resume
    dataset = args.dataset
    if dataset == 'sysu':
        data_path = '../SYSU-MM01/ori_data/'
        n_class = 395
        test_mode = [2, 1]
    elif dataset == 'regdb':
        data_path = '../Datasets/RegDB/'
        n_class = 206
        test_mode = [2, 1]
    elif dataset == 'nwpu':
        data_path = '../NWPU-ReID/ori_data/'
        n_class = 241
        if mode == 't2v':
            test_mode = [1, 2]  # thermal to visible
        if mode == 'v2t':
            test_mode = [2, 1]  # visible to thermal

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0
    pool_dim = 2048
    print('==> Building model..')
    if args.method == 'base':
        net = embed_net(n_class, no_local='off', gm_pool='off', arch=args.arch)
    else:
        net = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch)
    net.to(device)
    cudnn.benchmark = True

    checkpoint_path = args.model_path
    if mode == 't2v':
        checkpoint_path = 'save_model/'
    if mode == 'v2t':
        checkpoint_path = 'visible2thermal/'
    if args.method == 'id':
        criterion = nn.CrossEntropyLoss()
        criterion.to(device)

    print('==> Loading data..')
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop((args.img_h, args.img_w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.img_h, args.img_w)),
        transforms.ToTensor(),
        normalize,
    ])

    end = time.time()
    ########################################################################################################################
    print('==> Resuming from checkpoint..')
    if len(args.resume) > 0:
        model_path = checkpoint_path + args.resume
        # model_path = checkpoint_path + 'nwpu_awg_p4_n8_lr_0.1_seed_0_best.t'
        if os.path.isfile(model_path):
            print('==> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(model_path)
            net.load_state_dict(checkpoint['net'])
            print('==> loaded checkpoint {} (epoch {})'
                  .format(args.resume, checkpoint['epoch']))
        else:
            print('==> no checkpoint found at {}'.format(args.resume))

    # testing set
    query_img, query_label, query_cam = process_one_query_nwpu(pic_path)
    gall_img, gall_label, gall_cam = process_gallery_nwpu(data_path, mode=args.mode, trial=0, val=0, v2t=test_mode[0])

    nquery = len(query_label)
    ngall = len(gall_label)

    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False,
                                   num_workers=0)  # 调用TestData的getitem函数，然后按照batchsize打包

    query_feat_pool, query_feat_fc = extract_query_feat(query_loader, net, nquery, pool_dim, test_mode)

    gall_img, gall_label, gall_cam = process_gallery_nwpu(data_path, mode=args.mode, trial=0, val=0,
                                                          v2t=test_mode[0])

    trial_gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=0)
    if mode == 't2v':
        gall_feat_fc = np.load('gall_feat_fc_rgb.npy')
    elif mode == 'v2t':
        gall_feat_fc = np.load('gall_feat_fc_thermal.npy')
    distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
    target_gallery_path, labeltext, similartiytext = visualization(-distmat, query_label, gall_label, query_cam, gall_cam, gall_img, pic_path)  # result visualization
    return target_gallery_path, labeltext, similartiytext

if __name__ == '__main__':
    pic_path = r'D:\pythoncode\NWPU-ReID\ori_data\cam4\0124\0124_2022_03_13_15_21_45_4_0_0_526.png'
    target_gallery_path, labeltext, similartiytext = predict(pic_path, 'nwpu_agw_p4_n8_lr_0.1_seed_0_best.t', mode='t2v')
    print(target_gallery_path)
    print(labeltext)
    print(similartiytext)