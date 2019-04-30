import numpy as np
import os.path
import torch
import torch.optim as optim
from torch.nn import functional as F

from cpcvae import CPCVAE
from matplotlib import pyplot as plt
from tensorboard_logger import configure, log_value
from torch.autograd import Variable

def reset_grad(params):
    for p in params:
        if p.grad is not None:
            data = p.grad.data
            p.grad = Variable(data.new().resize_as_(data).zero_())

# Arguments
z_dim = 10
k_steps = 10
num_epochs = 1000
batch_size = 128
seed = 0
beta = 10

# Configure experiment path
savepath = 'z_%d_seed_%d_beta_%d' % (z_dim, seed, beta)
configure('%s/var_log' % savepath, flush_secs=5)

# Set seed
torch.manual_seed(seed)
np.random.seed(seed)

# Load data
data_file = 'data.npy'
data = np.load(data_file)
n_trajs = len(data)
print('Number of trajectories: %d' % n_trajs) #315
print('Number of transitions: %d' % sum([len(data[i]) for i in range(n_trajs)])) #19215

# Create CPC-VAE model
model = CPCVAE(z_dim=z_dim)
if torch.cuda.is_available():
    model.cuda()
solver = optim.RMSprop(list(model.parameters()), lr=1e-3)
params = list(model.parameters())

# Train model
for epoch in range(num_epochs):
    idx = np.random.choice(n_trajs, size=batch_size)
    if torch.cuda.is_available():
        idx = Variable(torch.from_numpy(idx)).cuda()
    for i in idx:
        t = np.random.randint(len(data[i]) - 1)
        o = data[i][t]
        o_next = data[i][t + 1]
        if torch.cuda.is_available():
            x = Variable(torch.cuda.FloatTensor(np.transpose(o, (2, 0, 1))[None]))
            x_next = Variable(torch.cuda.FloatTensor(np.transpose(o_next, (2, 0, 1))[None]))
        else:
            x = Variable(torch.FloatTensor(np.transpose(o, (2, 0, 1))[None]))
            x_next = Variable(torch.FloatTensor(np.transpose(o_next, (2, 0, 1))[None]))
        #CPC
        z, _ = model.encode(x)
        density_x = model.density(x_next, z)
        density_sum = 0
        for j in [n for n in range(n_trajs) if n != i]:
            k = np.random.randint(len(data[j]))
            o_other = data[j][k]
            if torch.cuda.is_available():
                x_other = Variable(torch.cuda.FloatTensor(np.transpose(o_other, (2, 0, 1))[None]))
            else:
                x_other = Variable(torch.FloatTensor(np.transpose(o_other, (2, 0, 1))[None]))
            density_sum += torch.exp(model.density(x_other, z) - density_x)
        density = 1.0 / (1.0 + density_sum)
        C_loss = -torch.mean(torch.log(density))
        #VAE
        xr, mu, logvar = model.forward(x)
        reconst_loss = F.binary_cross_entropy(xr, x / 255.0, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        V_loss = reconst_loss + beta * kl_div
        #0.5/0.5 weighting
        loss = 0.5 * C_loss + 0.5 * V_loss
        loss.backward()
        solver.step()
        reset_grad(params)

    print('********** Epoch %i ************' % epoch)
    print(loss)
    log_value('loss', loss, epoch)
    log_value('C_loss', C_loss, epoch)
    log_value('V_loss', V_loss, epoch)

    if not os.path.exists('%s/var' % savepath):
        os.makedirs('%s/var' % savepath)
    torch.save(model.state_dict(), '%s/var/cpcvae' % savepath)
