"""
YOLOv3网络结构分解：
    BackNone网络（Darknet53）->特征金字塔（FPN）->yolo head
模块划分：
    Input(batch_size*416*416*3)
        ||         /-DBL模块函数:conv+BN+LeakyRelu
    基础网络搭建—————**-Res*n模块函数：Res_blocks中包含有n个Res_units模块---（input->conv(1*1,stride=1)->conv(3*3,stride=1)->shortcut(+input)）
        ||         \-concat方式选择:维度拼接，会改变数据维度大小及尺寸
    三尺度输出层
    |   |   |
    y1  y2  y3
  大目标  中目标 小目标
（13*13*255） （26*26*255） （52*52*255）
"""
from collections import OrderedDict
import torch
import torch.nn as nn
import DarkNet53_set as Darknet53

darknet53=Darknet53.BlockNone()
# DBL模块函数
def DBL(filter_in,filter_out,kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    m=nn.Sequential(OrderedDict([
        ("conv",nn.Conv2d(filter_in, filter_out, kernel_size, stride=1, padding=pad,bias=False)),
        ("bn",nn.BatchNorm2d(filter_out)),
        ("relu",nn.LeakyReLU(0.1)),
    ]))
    return m

def DBL5(filter_list,filter_in,out_filter):
    m=nn.Sequential(
        DBL(filter_in,filter_list[0],1),
        DBL(filter_list[0],filter_list[1],3),
        DBL(filter_list[1],filter_list[0],1),
        DBL(filter_list[0], filter_list[1], 3),
        DBL(filter_list[1], filter_list[0], 1),
        # 获得预测结果卷积层
        DBL(filter_list[0], filter_list[1], 3),
        nn.Conv2d(filter_list[1],out_filter,kernel_size=1,stride=1,padding=0,bias=True)
    )
    return m


class YOLO(nn.Module):
    def __init__(self,num_classes,mask,pretrain=False):
        """
        :param num_classes: 检测物体种类数
        :param mask:先验框尺寸
        :param pretrain:是否使用预训练权重参数
        """
        super(YOLO, self).__init__()
        self.blocknone=darknet53
        if pretrain:
            # 添加使用预训练好的darknet53权重参数
            pass
        out_filter=[64,128,256,512,1024]
        # 开始搭建FPN
        self.layer0=DBL5([512,1024],out_filter[-1],len(mask[0])*(num_classes+5))

        self.layer1_conv=DBL(512,256,1)
        self.layer1_upsample=nn.Upsample(scale_factor=2,mode='nearest')
        self.layer1=DBL5([256,512],out_filter[-2]+256,len(mask[1])*(num_classes+5))

        self.layer2_conv = DBL(256,128, 1)
        self.layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.layer2 = DBL5([128,256], out_filter[-3] + 128, len(mask[2]) * (num_classes + 5))

    def forward(self,x):
        x2,x1,x0=self.blocknone(x)

        # 获取第一层预测特征(13,13,512)
        x0_DBL5=self.layer0[:5](x0)  # 取前五层作为卷积
        out1=self.layer0[5:](x0_DBL5) # 取后两层卷积作为y1预测层输出

        # 获取第二层预测特征(26,26,256)
        x1_in=self.layer1_conv(x0_DBL5)
        x1_in=self.layer1_upsample(x1_in)
        # concat
        x1_in=torch.cat([x1_in,x1],1)
        x1_DBL5=self.layer1[:5](x1_in)
        out2=self.layer1[5:](x1_DBL5)   # 第二层输出

        # 获取第三层的预测特征（52，52，128）
        x2_in=self.layer2_conv(x1_DBL5)
        x2_in=self.layer2_upsample(x2_in)
        # concat
        x2_in=torch.cat([x2_in,x2],1)
        out3=self.layer2(x2_in) # 第三层输出

        return out1,out2,out3
"""
每一个有效特征层将整个图片分成与其长宽对应的网格，如(N,13,13,255)的特征层就是将整个图像分成13x13个网格；
然后从每个网格中心建立多个先验框，这些框是网络预先设定好的框，网络的预测结果会判断这些框内是否包含物体，以及这个物体的种类。
"""

"""
m=YOLO(1,[[6, 7, 8], [3, 4, 5], [0, 1, 2]])
print(m)
from thop import profile            # 用来测试网络能够跑通，同时可查看FLOPs和params
input = torch.randn(1,3,256,256)     # 1张3通道尺寸为256x256的图片作为输入
flops, params = profile(m, (input,))
print(flops, params)
# 结果为:(1,18,8,8)(1,18,16,16)(,1,18,32,32)
"""