import numpy as np
import torch
from thop import profile, clever_format
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.net = nn.Sequential(
            # 为了和论文的图像输入尺寸保持一致以及下一层的55对应，这里对图像进行了padding
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=10e-4, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256 * 6 * 6, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=num_classes)
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.net:
            # 先一致初始化
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.normal_(layer.weight, mean=0, std=0.01) # 论文权重初始化策略
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 1)
            # 单独对论文网络中的2、4、5卷积层的偏置进行初始化
            nn.init.constant_(self.net[4].bias, 1)
            nn.init.constant_(self.net[10].bias, 1)
            nn.init.constant_(self.net[12].bias, 1)

    def forward(self, x):
        return self.net(x)

    def test_output_shape(self):
        test_img = torch.rand(size=(1, 3, 227, 227), dtype=torch.float32)
        for layer in self.net:
            test_img = layer(test_img)
            print(layer.__class__.__name__, 'output shape: \t', test_img.shape)


# 定义AlexNet网络模型
# MyLeNet5（子类）继承nn.Module（父类）
class MyAlexNet(nn.Module):
    # 子类继承中重新定义Module类的__init__()和forward()函数
    # init()：进行初始化，申明模型中各层的定义
    def __init__(self, num_classes=4):
        # super：引入父类的初始化方法给子类进行初始化
        super(MyAlexNet, self).__init__()
        # 卷积层，输入大小为224*224，输出大小为55*55，输入通道为3，输出为96，卷积核为11，步长为4
        self.c1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        # 使用ReLU作为激活函数
        self.ReLU = nn.ReLU()
        # MaxPool2d：最大池化操作
        # 最大池化层，输入大小为55*55，输出大小为27*27，输入通道为96，输出为96，池化核为3，步长为2
        self.s1 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 卷积层，输入大小为27*27，输出大小为27*27，输入通道为96，输出为256，卷积核为5，扩充边缘为2，步长为1
        self.c2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
        # 最大池化层，输入大小为27*27，输出大小为13*13，输入通道为256，输出为256，池化核为3，步长为2
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        # 卷积层，输入大小为13*13，输出大小为13*13，输入通道为256，输出为384，卷积核为3，扩充边缘为1，步长为1
        self.c3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 卷积层，输入大小为13*13，输出大小为13*13，输入通道为384，输出为384，卷积核为3，扩充边缘为1，步长为1
        self.c4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        # 卷积层，输入大小为13*13，输出大小为13*13，输入通道为384，输出为256，卷积核为3，扩充边缘为1，步长为1
        self.c5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        # 最大池化层，输入大小为13*13，输出大小为6*6，输入通道为256，输出为256，池化核为3，步长为2
        self.s5 = nn.MaxPool2d(kernel_size=3, stride=2)
        # Flatten()：将张量（多维数组）平坦化处理，神经网络中第0维表示的是batch_size，所以Flatten()默认从第二维开始平坦化
        self.flatten = nn.Flatten()
        # 全连接层
        # Linear（in_features，out_features）
        # in_features指的是[batch_size, size]中的size,即样本的大小
        # out_features指的是[batch_size，output_size]中的output_size，样本输出的维度大小，也代表了该全连接层的神经元个数
        self.f6 = nn.Linear(6*6*256, 4096)
        self.f7 = nn.Linear(4096, 4096)
        # 全连接层&softmax
        self.f8 = nn.Linear(4096, 1000)
        self.f9 = nn.Linear(1000, num_classes)

    # forward()：定义前向传播过程,描述了各层之间的连接关系
    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.s1(x)
        x = self.ReLU(self.c2(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.ReLU(self.c4(x))
        x = self.ReLU(self.c5(x))
        x = self.s5(x)
        x = self.flatten(x)
        x = self.f6(x)
         # Dropout：随机地将输入中50%的神经元激活设为0，即去掉了一些神经节点，防止过拟合
        # “失活的”神经元不再进行前向传播并且不参与反向传播，这个技术减少了复杂的神经元之间的相互影响
        x = F.dropout(x, p=0.5)
        x = self.f7(x)
        x = F.dropout(x, p=0.5)
        x = self.f8(x)
        x = F.dropout(x, p=0.5)
        x = self.f9(x)
        return x


if __name__ == '__main__':
    net = MyAlexNet(num_classes=4)
    # '''torchsummary 打印网络结构'''
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # net = net.to(device)
    # summary(net, (3, 244, 224), device=device)
    # '''计算模型的FLOPs'''
    # input = torch.randn(32, 3, 224, 224).cuda()
    # flops, params = profile(net, inputs=(input,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print(flops, params)
    model = net
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    dummy_input = torch.randn(32, 3, 224, 224, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings = np.zeros((repetitions, 1))
    # GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(mean_syn)