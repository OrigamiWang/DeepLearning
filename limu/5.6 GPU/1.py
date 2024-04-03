import torch 
from torch import nn

print(torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1'))

# 查询可用GPU数量
print(torch.cuda.device_count())

def try_gpu(i=0):
    if torch.cuda.device_count() >= i+1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(try_gpu(), try_gpu(10), try_all_gpus())


# tensor and GPU
x = torch.tensor([1, 2, 3])
print(x.device)

X = torch.ones(2, 3, device=try_gpu())
print(X)

Y = torch.rand(2, 3, device=try_gpu(1))
print(Y)




