# 1. padding 填充
# 解决原始图像边界信息丢失的问题
# 填充边缘0，输出会增加: (n_h - k_h + p_h + 1) * (n_w - k_w + p_w + 1)

# 填充p_h = k_h - 1, p_w = k_w - 1的好处
    # 1. 使得output_h == input_h, output_w == intput_w
    # 2. Y[i, j]是通过X[i, j]为中心，与卷积核进行cross correlation计算得到的
    
import torch 
from torch import nn

def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1)
X = torch.rand(size=(8, 8))
print(comp_conv2d(conv2d, X).shape)


conv2d = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
print(comp_conv2d(conv2d, X).shape)


# 2. stride步幅
# 卷积窗口可以跳过中间位置，每次滑动多个元素
# 为了计算的高效 / 缩减采样次数

# 输出形状：
# floor((n_h - k_h + p_h + s_h) / s_h) * floor((n_w - k_w + p_w + s_w) / s_w)
conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
print(comp_conv2d(conv2d, X).shape)