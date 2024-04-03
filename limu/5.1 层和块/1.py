import torch
from torch import nn
from torch.nn import functional as F

# 将以下代码 改写成块
# net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
# net(X)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256) # 隐藏层
        self.out = nn.Linear(256, 10) # 输出层
        
    # 定义函数的前向传播，即根据输入X，返回模型的最终输出
    def forward(self, X):
        # ReLU的函数版本，其在nn.functional模块中定义
        return self.out(F.relu(self.hidden(X)))
        
# net = MLP()
# print(net(X))


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # _modules 是一个OrderedDict()，保证了按照成员添加的顺序进行遍历
            self._modules[str(idx)] = module
    
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X


# net = MySequential(nn.Linear(20, 256),
#                    nn.ReLU(),
#                    nn.Linear(256, 10))
# print(net(X))


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随机权重参数。因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

# net = FixedHiddenMLP()
# print(net(X))

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

# chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
# print(chimera(X))