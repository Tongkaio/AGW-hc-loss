import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
from torch.autograd import Variable

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio  # 整除为1 比如，5//2=2

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        # torch.nn.init.constant(tensor, val)
        # 用val的值填充输入的张量或变量
        nn.init.constant_(self.W[1].weight, 0.0)  # 把31行 nn.BatchNorm2d(self.in_channels)的权重，置0
        nn.init.constant_(self.W[1].bias, 0.0)  # 把31行 nn.BatchNorm2d(self.in_channels)的偏置，置0


        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)  # torch.init.normal_：给tensor初始化，一般是给网络中参数weight初始化，初始化参数值符合正态分布。 https://blog.csdn.net/weixin_47156261/article/details/116902306
        init.zeros_(m.bias.data)  # 用常数0对偏置赋值 https://blog.csdn.net/weixin_46221946/article/details/122717592


def weights_init_classifier1(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:  # 如果输入的类是“Linear”返回0 则if为True，如果输入的类不是“Linear"返回-1，if为False
        init.normal_(m.weight.data, 0, 0.001)
        init.zeros_(m.bias.data)

def weights_init_classifier2(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:  # 如果输入的类是“Linear”返回0 则if为True，如果输入的类不是“Linear"返回-1，if为False
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)]
        feat_block += [nn.BatchNorm1d(low_dim)]

        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block

    def forward(self, x):
        x = self.feat_block(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]

        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier1)

        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输出特征图的大小1*1，通道数不发生变化，虽然叫自适应池化，但实际上就是全局池化
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

def before_Chunk_to_six(x):
    num_part = 6
    # pool size
    sx = x.size(2) / num_part
    sx = int(sx)
    kx = x.size(2) - sx * (num_part - 1)
    kx = int(kx)
    x = nn.functional.avg_pool2d(x, kernel_size=(kx, x.size(3)), stride=(sx, x.size(3)))
    # x = self.visible.avgpool(x)
    x = x.view(x.size(0), x.size(1), x.size(2))
    # x = self.dropout(x)
    return x

class embed_net(nn.Module):
    def __init__(self,  class_num, no_local='on', gm_pool='on', arch='resnet50', low_dim=512, drop=0.5):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        self.non_local = no_local

        pool_dim = 2048

        self.feature1 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature2 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature3 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature4 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature5 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.feature6 = FeatureBlock(pool_dim, low_dim, dropout=drop)
        self.classifier1 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier2 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier3 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier4 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier5 = ClassBlock(low_dim, class_num, dropout=drop)
        self.classifier6 = ClassBlock(low_dim, class_num, dropout=drop)

        self.l2norm = Normalize(2)
        if self.non_local == 'on':
            layers = [3, 4, 6, 3]
            non_layers = [0, 2, 2, 0]  # [0, 2, 3, 0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])  # 空列表 []
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift  # 不需要梯度

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)  # 对bottleneck使用weights_init_kaiming方法 https://blog.csdn.net/hxxjxw/article/details/119725864
        self.classifier.apply(weights_init_classifier2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool

    def forward(self, x1, x2, modal=0):
        if modal == 0:  # 双模态
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)  # 按维数0（行）拼接,x1在上，x2在下
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared block
        if self.non_local == 'on':
            NL1_counter = 0
            if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]  # 现在NL_1_idx = -1
            for i in range(len(self.base_resnet.base.layer1)):  # len(self.base_resnet.base.layer1)为3
                x = self.base_resnet.base.layer1[i](x)
                if i == self.NL_1_idx[NL1_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_1[NL1_counter](x)
                    NL1_counter += 1
            # Layer 2
            NL2_counter = 0
            if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]  # 这一步没有做 因为 NL_2_idx = [2, 3]
            for i in range(len(self.base_resnet.base.layer2)):
                x = self.base_resnet.base.layer2[i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x.shape  # _单个下划线表示无关紧要的变量，但又必须要有个东西来承接（本来这里是视频帧数，但我们是图片分类就不用了）
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1
            # Layer 3
            NL3_counter = 0
            if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.base.layer3)):
                x = self.base_resnet.base.layer3[i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            # Layer 4
            NL4_counter = 0
            if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet.base.layer4)):
                x = self.base_resnet.base.layer4[i](x)
                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1
        else:
            x = self.base_resnet(x)
        # when testing, modal will be 1 or 2, and HC LOSS will be useless, so are the relative net blocks.
        if modal == 0:  # when training, RGB and IR pictures will be load at the same time, but when testing is not
            x1, x2 = torch.chunk(x, 2, 0)
        if self.gm_pool  == 'on':
            b, c, h, w = x.shape  # torch.Size([batch, channel, h, w])
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)  # dim = -1 表示维度为从后往前数第一个维度，比如（B，C，A）中的A
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        feat = self.bottleneck(x_pool)
        if modal == 0:  # 双模态
            x1 = before_Chunk_to_six(x1)
            x1 = x1.chunk(6, 2)
            x1_0 = x1[0].contiguous().view(x1[0].size(0), -1)
            x1_1 = x1[1].contiguous().view(x1[1].size(0), -1)
            x1_2 = x1[2].contiguous().view(x1[2].size(0), -1)
            x1_3 = x1[3].contiguous().view(x1[3].size(0), -1)
            x1_4 = x1[4].contiguous().view(x1[4].size(0), -1)
            x1_5 = x1[5].contiguous().view(x1[5].size(0), -1)
            x2 = before_Chunk_to_six(x2)
            x2 = x2.chunk(6, 2)
            x2_0 = x2[0].contiguous().view(x2[0].size(0), -1)
            x2_1 = x2[1].contiguous().view(x2[1].size(0), -1)
            x2_2 = x2[2].contiguous().view(x2[2].size(0), -1)
            x2_3 = x2[3].contiguous().view(x2[3].size(0), -1)
            x2_4 = x2[4].contiguous().view(x2[4].size(0), -1)
            x2_5 = x2[5].contiguous().view(x2[5].size(0), -1)
            x_0 = torch.cat((x1_0, x2_0), 0)
            x_1 = torch.cat((x1_1, x2_1), 0)
            x_2 = torch.cat((x1_2, x2_2), 0)
            x_3 = torch.cat((x1_3, x2_3), 0)
            x_4 = torch.cat((x1_4, x2_4), 0)
            x_5 = torch.cat((x1_5, x2_5), 0)
            y_0 = self.feature1(x_0)
            y_1 = self.feature2(x_1)
            y_2 = self.feature3(x_2)
            y_3 = self.feature4(x_3)
            y_4 = self.feature5(x_4)
            y_5 = self.feature6(x_5)
            #y = self.feature(x)
            out_0 = self.classifier1(y_0)
            out_1 = self.classifier2(y_1)
            out_2 = self.classifier3(y_2)
            out_3 = self.classifier4(y_3)
            out_4 = self.classifier5(y_4)
            out_5 = self.classifier6(y_5)
            #out = self.classifier(y)
        if self.training:
            if modal == 0:  # 双模态
                return x_pool, self.classifier(feat), (out_0, out_1, out_2, out_3, out_4, out_5), (self.l2norm(y_0), self.l2norm(y_1), self.l2norm(y_2), self.l2norm(y_3), self.l2norm(y_4), self.l2norm(y_5))
            else:
                return x_pool, self.classifier(feat)
        else:
            if modal == 0:  # 双模态
                x_0 = self.l2norm(x_0)
                x_1 = self.l2norm(x_1)
                x_2 = self.l2norm(x_2)
                x_3 = self.l2norm(x_3)
                x_4 = self.l2norm(x_4)
                x_5 = self.l2norm(x_5)
                x = torch.cat((x_0, x_1, x_2, x_3, x_4, x_5), 1)
                y_0 = self.l2norm(y_0)
                y_1 = self.l2norm(y_1)
                y_2 = self.l2norm(y_2)
                y_3 = self.l2norm(y_3)
                y_4 = self.l2norm(y_4)
                y_5 = self.l2norm(y_5)
                y = torch.cat((y_0, y_1, y_2, y_3, y_4, y_5), 1)
                return self.l2norm(x_pool), self.l2norm(feat), x, y
            else:
                return self.l2norm(x_pool), self.l2norm(feat)


if __name__ == '__main__':
    net = embed_net(187)
    net.train()
    input = Variable(torch.FloatTensor(32, 3, 244, 188))
    x, y , out, norm = net(input, input)
