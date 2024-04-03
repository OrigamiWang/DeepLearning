import torch
from torch import nn

# nn.LazyLinear(256) 没有指定 输入的维度
# nn.LazyLinear(10) 没有指定 输入的维度
# 直到数据第一次通过模型传递时`net(X)`，框架才会动态地推断出每个层的大小
net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(),
                    nn.LazyLinear(10))

print(net)

X = torch.rand(2, 20)
net(X)
print(net)