from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData, NWPUData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb, eval_nwpu
from model import embed_net
from utils import *
from loss import OriTripletLoss, TripletLoss_WRT, CenterLoss
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from heterogeneity_loss import hetero_loss
# 命令行参数
parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='nwpu', help='dataset name: nwpu or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
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
parser.add_argument('--method', default='agw', type=str,
                    metavar='m', help='method type: base or agw')
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
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--thd', default=0, type=float,
                    help='threshold of Hetero-Center Loss')
parser.add_argument('--w_hc', default=0.5, type=float,
                    help='weight of Hetero-Center Loss')
parser.add_argument('--dist-type', default='l2', type=str,
                    help='type of distance')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)
w_hc = args.w_hc
dataset = args.dataset
if dataset == 'sysu':
    data_path = '../SYSU-MM01/ori_data/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [2, 1]  # visible to thermal
elif dataset == 'regdb':
    data_path = '../Datasets/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal
elif dataset == 'nwpu':
    data_path = '../NWPU-ReID/ori_data/'
    log_path = args.log_path + 'nwpu_log/'
    test_mode = [1, 2]  # thermal to visible

checkpoint_path = args.model_path  # 模型保存地址

if not os.path.isdir(log_path):  # 建立文件夹: 命令行日志、 保存的模型 、 Tensorboard里的图
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
if args.method == 'agw':  # 方法默认agw
    suffix = suffix + '_agw_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)
else:
    suffix = suffix + '_base_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)

# 优化器默认sgd 若不是sgd的话 如下处理
if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')  # 输出训练日志

vis_log_dir = args.vis_log_path + suffix + '/'  # Tensorboard的log文件

if not os.path.isdir(vis_log_dir):  # 建立文件夹
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))  # 打印展示所有args参数
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备选择cuda
best_acc = 0  # best test accuracy   # 每次epoch里的最高的正确率
start_epoch = 0  # epoch从0开始循环

print('==> Loading data..')  # 开始载入数据
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化 均值 标准差
transform_train = transforms.Compose([  # 对“训练集”  的变形操作
    transforms.ToPILImage(),  # 转为PILImage
    transforms.Pad(10),   # 填充
    transforms.RandomCrop((args.img_h, args.img_w)),  # 随机裁剪
    transforms.RandomHorizontalFlip(),  # 按照概率水平翻转
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([  # 对“测试集（这里其实为验证集）”  的变形操作
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()  # 开始载入数据 计时
if dataset == 'sysu':  # 数据集是sysu
    # training set
    trainset = SYSUData(data_path, transform=transform_train)  # 'SYSU-MM01/ori_data/' 2022 4月5日 19:51 看到此处
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)  # 2022 4月5日 22:33 看到此处

    # testing set  # 实际是验证集，都是红外图像，三个返回参数分别为: 图像路径，图像ID，相机ID
    # query是输入的待查询图像
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode, v2t=test_mode[1])  # 'SYSU-MM01/ori_data/' mode默认all
    # gall(gallery是搜索区域)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0, v2t=test_mode[0])  # 'SYSU-MM01/ori_data/'

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')  # trial默认为1
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')  # trial默认为1

elif dataset == 'nwpu':
    # training set
    trainset = NWPUData(data_path, transform=transform_train)  # 'SYSU-MM01/ori_data/' 2022 4月5日 19:51 看到此处
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)  # 2022 4月5日 22:33 看到此处

    # testing set  # 实际是验证集，都是红外图像，三个返回参数分别为: 图像路径，图像ID，相机ID
    # query是输入的待查询图像
    query_img, query_label, query_cam = process_query_nwpu(data_path, mode=args.mode, val=0, v2t=test_mode[1])  # 'SYSU-MM01/ori_data/' mode默认all
    # gall(gallery是搜索区域)
    gall_img, gall_label, gall_cam = process_gallery_nwpu(data_path, mode=args.mode, trial=0, val=0, v2t=test_mode[0])  # 'SYSU-MM01/ori_data/'

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))  # gallery图片集 numpy格式的
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))  # query图片集 numpy格式的

# testing data loader   这里用的是pytorch的DataLoader 载入验证集
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))  # 得到了trainset一共有多少个行人，即输出类别是多少，用了unique是因为train_color_label有重复的
nquery = len(query_label)  # 这里没用unique是因为要统计query里一共多少张
ngall = len(gall_label)  # 这里没用unique是因为要统计galley里一共多少张

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))  # RGB图像 类别数 张数
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))  # 红外图像 类别数 张数
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))  # query  类别数 张数
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))  # gallery  类别数 张数
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))  # 载入数据集花费的时间

##################################后面可以先注释掉，测试自己的数据集能否喂进去############################################

print('==> Building model..')
if args.method == 'base':  # method默认'agw', arch默认'resnet50'
    net = embed_net(n_class, no_local='off', gm_pool='off', arch=args.arch)
else:
    net = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch)  # 默认是选这个 都on  net放着模型，nclass是分类类别
net.to(device)  # 用GPU如果有的话
cudnn.benchmark = True  # 这一行可以加快卷积神经网络运行速度，具体见 https://zhuanlan.zhihu.com/p/73711222

if len(args.resume) > 0:  # 默认resume = '' 放着保存的模型，测试时用的，当前为训练，所以用不到
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion_id = nn.CrossEntropyLoss()  # 损失函数用交叉熵，这个是行人ID的损失
if args.method == 'agw':
    criterion_tri = TripletLoss_WRT()  # 三元组损失
else:
    loader_batch = args.batch_size * args.num_pos
    criterion_tri = OriTripletLoss(batch_size=loader_batch, margin=args.margin)

thd = args.thd
criterion_het = hetero_loss(margin=thd, dist_type=args.dist_type)  # dist_type默认L2  margin 0.3

criterion_het.to(device)
criterion_id.to(device)  # 用GPU
criterion_tri.to(device)  # 用GPU


if args.optim == 'sgd':  # 默认优化器为sgd梯度下降法，给不同的层设置不同的学习率
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                    + list(map(id, net.classifier.parameters())) \
                    + list(map(id, net.feature1.parameters())) \
                    + list(map(id, net.feature2.parameters())) \
                    + list(map(id, net.feature3.parameters())) \
                    + list(map(id, net.feature4.parameters())) \
                    + list(map(id, net.feature5.parameters())) \
                    + list(map(id, net.feature6.parameters())) \
                    + list(map(id, net.classifier1.parameters())) \
                    + list(map(id, net.classifier2.parameters())) \
                    + list(map(id, net.classifier3.parameters())) \
                    + list(map(id, net.classifier4.parameters())) \
                    + list(map(id, net.classifier5.parameters())) \
                    + list(map(id, net.classifier6.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())  # 去除掉了bottleneck和classifier的参数

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},  # 0.1的学习率
        {'params': net.bottleneck.parameters(), 'lr': args.lr},  # 输入的学习率
        {'params': net.classifier.parameters(), 'lr': args.lr},  # 输入的学习率
        {'params': net.feature1.parameters(), 'lr': args.lr},
        {'params': net.feature2.parameters(), 'lr': args.lr},
        {'params': net.feature3.parameters(), 'lr': args.lr},
        {'params': net.feature4.parameters(), 'lr': args.lr},
        {'params': net.feature5.parameters(), 'lr': args.lr},
        {'params': net.feature6.parameters(), 'lr': args.lr},
        {'params': net.classifier1.parameters(), 'lr': args.lr},
        {'params': net.classifier2.parameters(), 'lr': args.lr},
        {'params': net.classifier3.parameters(), 'lr': args.lr},
        {'params': net.classifier4.parameters(), 'lr': args.lr},
        {'params': net.classifier5.parameters(), 'lr': args.lr},
        {'params': net.classifier6.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)


# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def adjust_learning_rate(optimizer, epoch):  # 动态调整学习率  每30个epoch（训练轮数） 下降10%
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def train(epoch):

    current_lr = adjust_learning_rate(optimizer, epoch)  # 当前的学习率
    train_loss = AverageMeter()  # 这是一个具有 输出当前值val， 均值avg， 和sum，计数count 的类
    id_loss = AverageMeter()  # 类别损失值
    center_loss = AverageMeter()
    tri_loss = AverageMeter()  # 三元组损失值
    data_time = AverageMeter()  #
    batch_time = AverageMeter()

    correct = 0
    total = 0

    # switch to train mode
    net.train()  # 设置模型进入训练模式 如果模型中有Dropout和batchnorm等层，会起到作用 .val()也是一样
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):  # enumerate过程中实际上是dataloader按照其参数sampler规定的策略调用了其dataset的getitem方法。

        labels = torch.cat((label1, label2), 0)  # 按行拼接， label1在上，label2在下,但是因为label是一维的，所以还是按行（其实无行列的概念）拼接
        labels = Variable(labels.cuda().long())  # torch.Size([64])
        input1 = Variable(input1.cuda())  # Variable使得input具有了计算梯度 反向传播的能力# torch.Size([32, 3, 288, 144])
        input2 = Variable(input2.cuda())
        label1 = Variable(label1.cuda().long())
        label2 = Variable(label2.cuda().long())
        data_time.update(time.time() - end)

        # 4月16日，20：03
        outputs0, feat0, outputs, feat = net(input1, input2)  # x_pool, self.classifier(feat)  feat计算center loss + triplet loss , out0计算ID loss
        # labels = labels.to(torch.int64)  # 我加的，原因来自错误报告："nll_loss_forward_reduce_cuda_kernel_2d_index" not implemented for 'Int'
        loss_id = criterion_id(feat0, labels)  # label是ID号，color在上，红外在下
        loss_tri, batch_acc = criterion_tri(outputs0, labels)
        correct += (batch_acc / 2)
        _, predicted = feat0.max(1)  # max(1)是输出最大值的索引 size [64]
        correct += (predicted.eq(labels).sum().item() / 2)
        # *******************************************************************
        het_feat0 = feat[0].chunk(2, 0)
        het_feat1 = feat[1].chunk(2, 0)
        het_feat2 = feat[2].chunk(2, 0)
        het_feat3 = feat[3].chunk(2, 0)
        het_feat4 = feat[4].chunk(2, 0)
        het_feat5 = feat[5].chunk(2, 0)
        loss_c0 = criterion_het(het_feat0[0], het_feat0[1], label1, label2)
        loss_c1 = criterion_het(het_feat1[0], het_feat1[1], label1, label2)
        loss_c2 = criterion_het(het_feat2[0], het_feat2[1], label1, label2)
        loss_c3 = criterion_het(het_feat3[0], het_feat3[1], label1, label2)
        loss_c4 = criterion_het(het_feat4[0], het_feat4[1], label1, label2)
        loss_c5 = criterion_het(het_feat5[0], het_feat5[1], label1, label2)
        loss0 = w_hc * loss_c0
        loss1 = w_hc * loss_c1
        loss2 = w_hc * loss_c2
        loss3 = w_hc * loss_c3
        loss4 = w_hc * loss_c4
        loss5 = w_hc * loss_c5
        # *******************************************************************
        loss_center_avg = (loss0 + loss1 + loss2 + loss3 + loss4 + loss5) / 3
        loss = loss_id + loss_tri + loss_center_avg
        optimizer.zero_grad()  # 进行下一次batch梯度计算的时候，前一个batch的梯度计算结果，没有保留的必要了 https://blog.csdn.net/bigbigvegetable/article/details/114674793

        # torch.autograd.backward([loss0, loss1, loss2, loss3, loss4, loss5],
        #                         [torch.tensor(1.0).cuda(), torch.tensor(1.0).cuda(), torch.tensor(1.0).cuda(),
        #                          torch.tensor(1.0).cuda(), torch.tensor(1.0).cuda(), torch.tensor(1.0).cuda()],
        #                         retain_graph=True)
        loss.backward()
        optimizer.step()  # optimizer.step用来更新参数，就是图片中下半部分的w和b的参数更新操作 https://blog.csdn.net/bigbigvegetable/article/details/114674793

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        # print(loss_center_avg, loss_center_avg.type)
        if torch.is_tensor(loss_center_avg):
            center_loss.update(loss_center_avg.item(), 2 * input1.size(0))
        else:
            center_loss.update(loss_center_avg, 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'CLoss: {center_loss.val:.4f} ({center_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss, center_loss=center_loss))

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)


def test(epoch):  #
    # switch to evaluation mode
    net.eval()  # 进入验证模式
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))  # ngall 为 gallery张数
    gall_feat_att = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):  # gall_loader是验证集的gallery图
            batch_num = input.size(0)  # batch_
            input = Variable(input.cuda())  # 输入图像
            feat, feat_att, _, _ = net(input, input, test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()  # detach意思为不要计算梯度， numpy 是转为numpy
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    query_feat_att = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att, _, _ = net(input, input, test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))  # 用内积计算”余弦相似度”
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP  = eval_regdb(-distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att  = eval_regdb(-distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
    elif dataset == 'nwpu':
        cmc, mAP, mINP = eval_nwpu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_nwpu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    writer.add_scalar('rank1_att', cmc_att[0], epoch)
    writer.add_scalar('mAP_att', mAP_att, epoch)
    writer.add_scalar('mINP_att', mINP_att, epoch)
    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att


# training
print('==> Start Training...')
for epoch in range(start_epoch, 81 - start_epoch):  # 81轮

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)  # RGB图的索引
    print(trainset.tIndex)  # 红外图的索引

    loader_batch = args.batch_size * args.num_pos  #  8 * 4

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training

    train(epoch)

    if epoch > 0 and epoch % 2 == 0:  # 每2轮进行一次验证
        print('Test Epoch: {}'.format(epoch))

        # testing
        cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch)
        # save model
        if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc_att[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc_att,
                'mAP': mAP_att,
                'mINP': mINP_att,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        # save model
        if epoch > 10 and epoch % args.save_epoch == 0:
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print('Best Epoch [{}]'.format(best_epoch))


