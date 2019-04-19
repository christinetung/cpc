import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from blockwall.utils import stochastic_binary_layer, from_numpy_to_var

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)

class CPC(nn.Module):
    def __init__(self, z_dim, output_type="continuous"):
        super(CPC, self).__init__()
        self.encoder = nn.Sequential(
            # 3 x 64 x 64
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 x 31 x 31
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 14 x 14
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 6 x 6
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 2 x 2
            Flatten()
        )
        self.fc1 = nn.Linear(1024, z_dim)
        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def encode(self, x):
        z = self.encoder(x)
        z = self.fc1(z)
        if self.output_type == "binary":
            z_out = stochastic_binary_layer(z)
        elif self.output_type == "onehot":
            z_out = F.gumbel_softmax(z, hard=True)
        else:
            assert self.output_type == "continuous"
            z_out = z
        return z_out

    def density(self, x_next, z):
        #TODO: @Christine I wanted to rename this log_density but I don't want to overwrite your function so I created another one.
        batch_size = z.shape[0]
        z_next = self.encode(x_next)
        z_next = z_next.unsqueeze(2)
        z = z.unsqueeze(2)
        w = self.W.repeat(batch_size, 1, 1)
        f_out = torch.bmm(torch.bmm(z_next.permute(0, 2, 1), w), z)
        f_out = f_out.squeeze()
        return f_out

    def log_density(self, x_next, z):
        # Same as density
        assert x_next.size(0) == z.size(0)
        z_next = self.encode(x_next)
        z_next = z_next.unsqueeze(2)
        z = z.unsqueeze(2)
        w = self.W.repeat(z.size(0), 1, 1)
        f_out = torch.bmm(torch.bmm(z_next.permute(0, 2, 1), w), z)
        f_out = f_out.squeeze()
        return f_out

class VAE(nn.Module):
    def __init__(self, z_dim, h_dim=1024, output_type="continuous"):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            # 3 x 64 x 64
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 x 31 x 31
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 14 x 14
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 6 x 6
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 2 x 2
            Flatten(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            UnFlatten(),
            # 1024 x 1 x 1
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # 128 x 5 x 5
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # 64 x 13 x 13
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 32 x 30 x 30
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
            nn.Sigmoid(),
            # 3 x 64 x 64
        )
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.output_type = output_type

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = from_numpy_to_var(np.random.randn(*mu.size()))
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        if self.output_type == "binary":
            z_out = stochastic_binary_layer(z)
        elif self.output_type == "onehot":
            z_out = F.gumbel_softmax(z, hard=True)
        else:
            assert self.output_type == "continuous"
            z_out = z
        return z_out, mu, logvar

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        o = self.decode(z)
        return o, mu, logvar

    def log_density(self, x_next, z):
        # Same as density
        assert x_next.size(0) == z.size(0)
        z_next, _, _ = self.encode(x_next)
        z_next = z_next.unsqueeze(2)
        z = z.unsqueeze(2)
        w = self.W.repeat(z.size(0), 1, 1)
        f_out = torch.bmm(torch.bmm(z_next.permute(0, 2, 1), w), z)
        f_out = f_out.squeeze()
        return f_out



def loss_function(recon_x, x, mu, logvar, beta=1):
    # import ipdb
    # ipdb.set_trace()
    BCE = F.binary_cross_entropy(recon_x.view(-1, 64*64), x.view(-1, 64*64), size_average=False)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return (BCE + beta*KLD)/x.size(0)

def variational_lower_bound(recon_x, x, mu, logvar):
    input = recon_x.view(-1, 3 * 64 * 64)
    target = x.view(-1, 3 * 64 * 64)
    BCE = target * torch.log(input) + (1 - target) * torch.log(1 - input)
    KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return BCE.sum(1), KLD.sum(1)

class Decoder(nn.Module):
    def __init__(self, h_dim=1024, z_dim=10):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def decode(self, z):
        x = self.fc1(z)
        x = self.decoder(x)
        return x
