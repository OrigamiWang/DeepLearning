import torch 
from torch import nn
from torch.nn import functional as F

# 不带参数的层
class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()
    

layer = CenteredLayer()
# print(layer(torch.arange(1, 6, dtype=torch.float32)))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8)) # Y ∈ R^(4*128) 
# print(Y)


# 带参数的层
class MyLinear(nn.Module):
    # unit in, unit out 
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))

    def forward(self, X):
        # Y = XW + b
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        # ReLU激活
        return F.relu(linear)

linear = MyLinear(5, 3)
# print(linear.weight)
print(linear(torch.rand(2, 5)))
