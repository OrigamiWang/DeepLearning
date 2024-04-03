import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
y = torch.zeros(4)
# 1. 保存/加载 tensor
# save in sencondary file 
# torch.save(x, 'x-file')

# load file on memory
# x2 = torch.load('x-file')
# print(x2)

# mydict = {'x': x, 'y': y}
# torch.save(mydict, 'mydict')
# mydict2 = torch.load('mydict')
# print(mydict2)

# 保存/加载 模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

# save model params in file
# state_dict: 返回包含整个net的params dict, 包括weight、bias...
# 用于将model 序列化，将这个model的params保存、加载到另一个model中
torch.save(net.state_dict(), 'mlp.params')
print(net.state_dict())

# load/recover model

clone = MLP()
clone.load_state_dict(torch.load('mlp.params'))
# eval：将model设置为评估模式
# 当加载一个模型并准备进行预测时，应该总是调用.eval()来确保模型处于评估模式。这样可以确保模型的行为与训练时不同，更适合于评估和测试。
# 禁用dropout层、保持params不被改变
print(clone.eval())


Y_clone = clone(X)
# Y = net(X)
print(Y_clone == Y)


# Q: 如何只复用网络的一部分
torch.save(net.hidden.state_dict(), 'mlp.hidden.params')
clone = MLP()
clone.hidden.load_state_dict(torch.load('mlp.hidden.params'))
print(clone.hidden.weight == net.hidden.weight)