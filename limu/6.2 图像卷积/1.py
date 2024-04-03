import torch
from torch import nn
from d2l import torch as d2l

# 卷积神经网络CNN（convolutional neural networks）

# 1. 互相关运算 cross correlation
def corr2d(X, K):
    """计算二维互相关运算"""
    # 输出大小Y等于输入大小X减去卷积核大小K
    # (n_h - k_h + 1) * (n_w - k_w + 1)
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # cross correlation其实就是按元素乘法再求和
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))


# 2. 卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        # 随机初始化卷积核权重
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
    

# 3. 图像中目标的边缘检测
X = torch.ones(size=(6, 8))
X[:, 2:6] = 0
print(X)

# 1*2卷积核
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
# 输出Y中的1代表从白色到黑色的边缘，-1代表从黑色到白色的边缘，其他情况的输出为0
print(Y)

# 这个卷积核K只可以检测垂直边缘，无法检测水平边缘
print(corr2d(X.t(), K))


# 4. 学习卷积核
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

# 批量大小、通道、高度、宽度
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))
lr = 3e-2

for i in range(10):
     Y_hat = conv2d(X)
     l = (Y_hat - Y) ** 2
     conv2d.zero_grad()
     l.sum().backward()
     # [:]是一种就地操作，-=也是就地操作，此时用不用[:]没有区别
     conv2d.weight.data[:] -= lr * conv2d.weight.grad
     if (i + 1) % 2 == 0:
        print(f'epoch {i+1}, loss {l.sum():.3f}')
        
        

print(conv2d.weight.data.reshape((1, 2)))

# 卷积核张量上的权重，我们称其为元素。

# 感受野（receptive field）是指在前向传播期间可能影响
# x计算的所有元素（来自所有先前层）。


X = torch.eye(8)
print(X)
print(corr2d(X, K))
print(corr2d(X.t(), K))
print(corr2d(X, K.t()))