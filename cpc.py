import torch
import torch.nn as nn
import torch.nn.functional as F

class CPC(nn.Module):
    """
    CPC
    """

    def __init__(self, x_dim, z_dim, batch_size, hidden=(100, 100), tau=1.0):
        super(CPC, self).__init__()
        self.batch_size = batch_size
        self.tau = tau

        self.efc1 = nn.Linear(x_dim, hidden[0])
        self.efc2 = nn.Linear(hidden[0], hidden[1])
        self.efc3 = nn.Linear(hidden[1], z_dim)

        self.pfc1 = nn.Linear(z_dim, hidden[0])
        self.pfc2 = nn.Linear(hidden[0], hidden[1])
        self.pfc3 = nn.Linear(hidden[1], x_dim)

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))

    def encode(self, x):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, length 10 one hot vector, ex. [1 0 0 0 0 0 0 0 0 0]
        """
        _ = torch.tanh(self.efc1(x))
        _ = torch.tanh(self.efc2(_))
        z_out = F.gumbel_softmax(self.efc3(_), tau=self.tau, hard=True)
        return z_out

    def get_dist(self, x):
        """
        TODO: fill in
        :param x: x_t, x y coordinates
        :return: z_t, length 10 softmax result
        """
        _ = torch.tanh(self.efc1(x))
        _ = torch.tanh(self.efc2(_))
        z_out = F.softmax(self.efc3(_), dim=1)
        return z_out

    def predict(self, z):
        """
        Prediction: x_t+k = p(z_t)
        :param z: z_t
        :return: x_t+1
        """
        _ = torch.tanh(self.pfc1(z))
        _ = torch.tanh(self.pfc2(_))
        x_next_out = self.pfc3()
        return x_next_out

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
        w = torch.abs(self.W.repeat(self.batch_size, 1, 1))
        f_out = torch.bmm(torch.bmm(z_next.permute(0, 2, 1), w), z).squeeze()
        return f_out

'''
class Encoder(nn.Module):
    """
    Nonlinear encoder Genc maps the input sequence of
    observations xt to a sequence of latent representations
    zt = Genc(xt).
    """

    def __init__(self, x_dim, z_dim, hidden=(100, 100)):
        super(Genc, self.__init__())
        self.conv1 = nn.Conv2d(x_dim, hidden[0], 8, stride=4)
        self.conv2 = nn.Conv2d(hidden[0], hidden[1], 4, stride=2)
        self.fc = nn.Linear(hidden[1], z_dim)

    def forward(self, x):
        z_out = F.tanh(self.conv1(x))
        z_out = F.tanh(self.conv2(z_out))
        z_out = self.fc(z_out)
        return z_out

class Generator(nn.Module):
    """
    generative model
    pk(xt+k|ct).
    """

    def __init__(self, x_dim, c_dim, hidden=(50, 50)):
        super(Generator, self.__init__())
        self.fc1 = nn.Linear(2 * c_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3_1 = nn.Linear(hidden[1], x_dim)
        self.fc3_2 = nn.Linear(hidden[1], x_dim)

    def forward(self, c, c_next):
        input = torch.cat([c, c_next], dim=1)
        G_h1 = F.tanh(self.fc1(input))
        G_h2 = F.tanh(self.fc2(G_h1))
        G_out = self.fc3_1(G_h2)
        G_next = self.fc3_2(G_h2)
        return G_out, G_next

    def get_prob(self, c):
        G_h1 = F.tanh(self.fc1(c))
        G_h2 = F.tanh(self.fc2(G_h1))
        G_out = self.fc3_1(G_h2)
        G_prob = F.softmax(G_out, dim=1)
        return G_prob

class Density(nn.Module):
    """
    log-bilinear model
    fk(xt+k, ct) = exp(zWkct)
    """

    def __init__(self, x_dim, c_dim):
        super(Density, self).__init()
        self.fc1 = nn.Linear(x_dim + c_dim, 1, bias=False)

    def get_density(self, x, c):
        input = torch.cat([x, c], dim=1)
        return torch.exp(self.fc1(input))


class Posterior(nn.Module):
    """
    p(ct|xt)
    """

    def __init__(self, x_dim, c_dim, hidden=(100, 100)):
        super(Posterior, self).__init__()
        self.c_dim = c_dim
        self.fc1 = nn.Linear(x_dim, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], c_dim)

    def forward(self, x):
        _ = F.tanh(self.fc1(x))
        _ = F.tanh(self.fc2(_))
        return F.softmax(self.fc3(_), dim=1)

class Transition(nn.Module):
    """
    p(ct+k|ct)
    """

    def __init__(self, c_dim, tau=1.0):
        super(Transition, self).__init__()
        self.fc1 = nn.Linear(c_dim, c_dim, bias=False)
        self.tau = tau

    def forward(self, c):
        c_next = F.gumbel_softmax(self.fc1(c), tau=self.tau, hard=True)
        return c_next

    def get_prob(self, c):
        T_prob = F.softmax(self.fc1(c), dim=1)
        return T_prob
'''