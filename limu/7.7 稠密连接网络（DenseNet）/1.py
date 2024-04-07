# DenseNet和ResNet比较
# ResNet：相加 -- f(x) = x + g(x) 
# DenseNet：稠密连接 -- x -> [x, f1(x), f2([x, f1(x)]), f3([x, f1(x), f2([x, f1(x)])])]
# 稠密连接：最后一层和之前的所有层紧密连接
# 稠密网络 = 稠密块（dense block）+ 过渡层（transition layer）


import torch
from torch import nn
from d2l import torch as d2l

# 稠密块
def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
    )

class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            # input_channels = num_channels * i + input_channels
            # 因为在一个稠密块（Dense Block）中，
            # 每个卷积层的输出都会与其输入在通道维度上进行拼接
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels
            ))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X
    

blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
# torch.Size([4, 23, 8, 8])
print(Y.shape)

# 过渡层
def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        # 减小通道数
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        # 宽高减半
        nn.AvgPool2d(kernel_size=2, stride=2)
    )

blk = transition_block(23, 10)
# torch.Size([4, 10, 4, 4])
print(blk(Y).shape)

# DenseNet模型
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

num_channels, growth_rate = 64, 32
# 每个稠密块使用多少个卷积层
num_convs_in_dense_blocks = [4, 4, 4, 4]
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 加上上一个稠密块的输出通道数
    num_channels += num_convs * growth_rate
    if i != len(num_convs_in_dense_blocks) - 1:
        # 使用过渡层来减半高和宽
        blks.append(transition_block(num_channels, num_channels // 2))
        # 减半通道数
        num_channels = num_channels // 2


net = nn.Sequential(
    b1, *blks,
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(),
    nn.Linear(num_channels, 10))


lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())