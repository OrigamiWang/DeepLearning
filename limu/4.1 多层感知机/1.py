import torch
from d2l import torch as d2l
from util import p

# 常见的激活函数(activation function)

# 1. ReLU (rectified linear unit) 修正线性单元
# ReLU(x) = max(x, 0)
# 当输入为负时，ReLU函数的导数为0
# 而当输入为正时，ReLU函数的导数为1
# 输入为0，不可导
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
d2l.plt.show()

y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
d2l.plt.show()

# pReLU (paramaterized ReLU)
# pReLU(x) = max(x, 0) + αmin(0, x)

# 2. sigmoid函数 --- 挤压函数
# (-inf, inf) -> (0, 1)
# sigmoid(x) = 1 / (1 + exp(-x))
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
d2l.plt.show()

x.grad.data.zero_()
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
d2l.plt.show()

# 3. tanh函数 --- 双曲正切函数
# tanh(x) = (1 - exp(-2x)) / (1 + exp(-2x))
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
d2l.plt.show()
p(y)
x.grad.data.zero_()
# 如果y是标量，那么不需要y.backward()不需要参数
# 如果y是张量，那么y.backward()需要参数
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
d2l.plt.show()
p(torch.ones_like(x))