import torch
import torch.nn as nn
import torch.nn.functional as F
from blockwall.utils import stochastic_binary_layer

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
            # 32 x 32 x 32
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 128 x 8 x 8
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 256 x 4 x 4
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
