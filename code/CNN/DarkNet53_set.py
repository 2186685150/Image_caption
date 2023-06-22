
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
import math
from collections import OrderedDict
import torch.nn as nn

# DBL模块函数
def DBL(filter_in,filter_out,kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    m=nn.Sequential(OrderedDict([
        ("conv",nn.Conv2d(filter_in, filter_out, kernel_size, stride=1, padding=pad,bias=False)),
        ("bn",nn.BatchNorm2d(filter_out)),
        ("relu",nn.LeakyReLU(0.1)),
    ]))
    return m

# ——————————————————————————————#
# 搭建Darknet53网络
# 残差结构使用类实现
# ——————————————————————————————#

# 残差模块类
class Res_Units(nn.Module):
    def __init__(self,input_filter,filter_in_list):
        super(Res_Units,self).__init__()
        self.convolutional0=DBL(input_filter,filter_in_list[0],1)
        self.convolutional1=DBL(filter_in_list[0],filter_in_list[1],3)

    def forward(self,x):
        res=x

        out=self.convolutional0(x)
        out=self.convolutional1(out)
        out+=res
        return out

class BlockNone(nn.Module):
    def __init__(self):
        super (BlockNone,self).__init__()
        self.input=32
        # 输入层
        # (256,256,3)->(256,256,32)
        self.conv0=nn.Conv2d(3,self.input,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn0=nn.BatchNorm2d(self.input)
        self.relu=nn.LeakyReLU(0.1)

        # 设计Res_Block
        self.block1=self.make_blocks([32,64],1)
        self.block2=self.make_blocks([64,128],2)
        self.block3 = self.make_blocks([128,256], 8)
        self.block4 = self.make_blocks([256,512], 8)
        self.block5 = self.make_blocks([512,1024], 4)


        # 初始化权值
        for i in self.modules():
            if isinstance(i,nn.Conv2d):
                n=i.kernel_size[0]*i.kernel_size[1]*i.out_channels
                i.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(i,nn.BatchNorm2d):
                i.weight.data.fill_(1)
                i.bias.data.zero_()

    def make_blocks(self,filters,n):
        layers=[]
        # 向下采样，然后堆叠res_units
        layers.append(("res_conv",nn.Conv2d(self.input,filters[1],kernel_size=3,stride=2,padding=1,bias=False)))
        layers.append(('res_bn',nn.BatchNorm2d(filters[1])))
        layers.append(('res_relu',nn.LeakyReLU(0.1)))
        self.input=filters[1]
        # 堆叠res_units
        for i in range(0,n):
            layers.append(("resdual_{}".format(i),Res_Units(self.input,filters)))
        return nn.Sequential(OrderedDict(layers))

    def forward(self,x):
        x=self.conv0(x) # input(batch_size,3,416,416)--->(batch_size,32,416,416)
        x=self.bn0(x)
        x=self.relu(x)

        x=self.block1(x)    # input(batch_size,32,416,416)--->(batch_size,64,208,208)
        x=self.block2(x)    # input(batch_size,64,208,208)--->(batch_size,128,104,104)
        y1=self.block3(x)   # output(256,52,52)
        y2=self.block4(y1)  # output(512,26,26)
        y3=self.block5(y2)  # output(1024,13,13)

        return y1,y2,y3

def darknet53():
    model=BlockNone()
    return model
