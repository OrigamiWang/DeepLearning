import torch 
from torch import nn
from d2l import torch as d2l

# load data
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# set model params
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))
params = [W1, b1, W2, b2]

# activate function
def ReLU(X):
    # ReLU(x) = max(x, 0)
    a = torch.zeros_like(X)
    return torch.max(X, a)

# model
def net(X):
    X = X.reshape((-1, num_inputs))
    # H = X@W1 + b1
    # O = ReLU(H@W2 + b2)
    H = torch.matmul(X, W1) + b1
    O = ReLU(torch.matmul(H, W2) + b2)
    return O

# loss function
# reduction='none: 返回与input同形状的张量，为每个sample的loss
loss = nn.CrossEntropyLoss(reduction='none')

# train
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
d2l.plt.show()

d2l.predict_ch3(net, test_iter)
d2l.plt.show()