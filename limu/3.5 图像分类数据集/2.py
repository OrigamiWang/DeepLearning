import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from util import p

import matplotlib.pyplot as plt
import time
import numpy as np

def get_dataloader_workers():  #@save
    """使用4个进程来读取数据"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):  #@save
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

if __name__ == '__main__':             
    batch_sizes = [2**i for i in range(1, 10)]  # 2, 4, 8, ..., 512
    load_times = []

    for batch_size in batch_sizes:
        start_time = time.time()
        train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=64)
        # 这里我们只是初始化了 DataLoader，为了实际测量加载时间，我们需要迭代数据集
        for X, y in train_iter:
            pass  # 这里可以是实际的处理逻辑，但现在我们只是通过它来模拟数据加载
        end_time = time.time()
        load_times.append(end_time - start_time)  # 确保每次迭代都记录时间

    # 确保 load_times 的长度与 batch_sizes 的长度相同
    assert len(load_times) == len(batch_sizes), "The length of load_times and batch_sizes must be the same."

    # 绘制对数批量大小与加载时间的关系图
    plt.plot(np.log2(batch_sizes), load_times, marker='o')
    plt.xlabel('Log2 of Batch Size')
    plt.ylabel('Time (seconds)')
    plt.title('Loading Time vs. Log2 of Batch Size')
    plt.show()