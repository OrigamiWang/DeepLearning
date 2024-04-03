import torch
from d2l import torch as d2l

# sigmoid 只有当输入接近于0时，才比较稳定
# 如果输入过大或者过小都会导致 --> 梯度消失(gradient vanishing)
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
# d2l.plt.show()

# 梯度爆炸(gradient exploding)
M = torch.normal(0, 1, size=(4, 4))
print(f'a matrix: {M}')
for i in range(100):
    # torch.mm只能处理两个二位张量（矩阵）乘法
    M = torch.mm(M, torch.normal(0, 1, size=(4, 4)))

print(f'after mutiple 100 matrix: {M}')


# Xavier初始化 ---> 用于对weight进行初始化（例如用正态分布）

