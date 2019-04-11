import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

def from_numpy_to_var(npx, dtype='float32'):
    var = Variable(torch.from_numpy(npx.astype(dtype)))
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var

def reset_grad(params):
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())

def binary_to_int(x, width):
    return x.dot(2 ** np.arange(width)[::-1])

def stochastic_binary_layer(x, tau=1.0):
    """
    x is (batch size)x(N) input of binary logits. Output is stochastic sigmoid, applied element-wise to the logits.
    """
    orig_shape = list(x.size())
    x0 = torch.zeros_like(x)
    x = torch.stack([x, x0], dim=2)
    x_flat = x.view(-1, 2)
    out = F.gumbel_softmax(x_flat, tau=tau, hard=True)[:, 0]
    return out.view(orig_shape)

def onehot_to_int(x):
    return np.where(x==1)[1]