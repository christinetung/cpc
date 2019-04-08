import torch
import torch.nn as nn
import torch.nn.functional as F
from cpc_utils import stochastic_binary_layer

class CPC(nn.Module):
    """
    CPC
    """

    def __init__(self, x_dim, z_dim, batch_size, hidden=(100, 100), output_type="continuous"):
        super(CPC, self).__init__()
        self.batch_size = batch_size

        self.efc1 = nn.Linear(x_dim, hidden[0])
        self.efc2 = nn.Linear(hidden[0], hidden[1])
        self.efc3 = nn.Linear(hidden[1], z_dim)

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type
    def encode(self, x):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        _ = torch.tanh(self.efc1(x))
        _ = torch.tanh(self.efc2(_))
        z_out = self.efc3(_)
        if self.output_type == "binary":
            z_out = stochastic_binary_layer(z_out)
        elif self.output_type == "onehot":
            z_out = F.gumbel_softmax(z_out, hard=True)
        return z_out

    def density(self, x_next, z):
        """
        Density: f(x_t+k, z_t) prop exp(z_t+k^T W_k z_t)
        :param x_next: x_t+1
        :param z: z_t
        :return: f(x_t+1, z_t)
        """
        z_next = self.encode(x_next)
        z_next = z_next.unsqueeze(2)
        z = z.unsqueeze(2)
        w = self.W.repeat(self.batch_size, 1, 1)
        f_out = torch.exp(torch.bmm(torch.bmm(z_next.permute(0, 2, 1), w), z))
        f_out = f_out.squeeze()
        return f_out

    def log_density(self, x_next, z):
        assert x_next.size(0) == z.size(0)
        z_next = self.encode(x_next)
        z_next = z_next.unsqueeze(2)
        z = z.unsqueeze(2)
        w = self.W.repeat(z.size(0), 1, 1)
        f_out = torch.bmm(torch.bmm(z_next.permute(0, 2, 1), w), z)
        f_out = f_out.squeeze()
        return f_out

