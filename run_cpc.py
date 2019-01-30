import torch
import torch.optim as optim
import numpy as np
import os.path

from matplotlib import pyplot as plt, cm, colors, gridspec
from tensorboard_logger import configure, log_value, log_histogram
from torch.autograd import Variable
from cpc import CPC
from cpc_utils import from_numpy_to_var, reset_grad, get_map, plot_data_density, plot_map

# Arguments
x_dim = 2
z_dim = 10
batch_size = 256
data_size = batch_size * 150
k_steps = 5
horizon = 60
mini_data_size = int(data_size / k_steps / horizon)
seed = 42

assert mini_data_size * k_steps * horizon == data_size

# Configure experiment path
savepath = "filter42"
configure("%s/var_log" % savepath, flush_secs=5)

# Set seed
torch.manual_seed(seed)

# Generate no obstacle map
map2d = get_map('no')

fig = plt.figure(figsize=(8, 12))
gs = gridspec.GridSpec(3, 2)
plt.subplot(gs[0, 0])
trajs = np.zeros((data_size, 2 * x_dim))
for i in range(mini_data_size):
    map2d.reset()
    _x = np.zeros(horizon + k_steps)
    _y = np.zeros(horizon + k_steps)
    for t in range(horizon + k_steps):
        _x[t], _y[t] = map2d.step()
    if i < 5:
        cmap = cm.get_cmap('jet')
        plt.plot(_x, _y, c=cmap(i / 5), zorder=-1)
    for j in range(1, k_steps + 1):
        trajs[i * horizon * k_steps + horizon * (j - 1): i * horizon * k_steps + horizon * j] = \
            np.array([_x[:horizon], _y[:horizon],
                      _x[j:j + horizon], _y[j:j + horizon]]).T
plot_map(map2d)

plt.subplot(gs[1:3, :])
plot_map(map2d)
plot_data_density(trajs[:6000, :x_dim])

plt.savefig('%s/data.png' % savepath)
plt.close()

data = from_numpy_to_var(trajs)
print('Data size: %d' % (data_size))

# Create CPC model
C = CPC(x_dim, z_dim, batch_size)
if torch.cuda.is_available():
    C.cuda()

C_solver = optim.RMSprop(list(C.parameters()), lr=1e-3)
params = list(C.parameters())

# Train CPC model
for epoch in range(1000):
    n_batch = int(data_size / batch_size)
    for it in range(n_batch):
        idx = np.random.choice(data_size, size=batch_size)
        if torch.cuda.is_available():
            idx = Variable(torch.from_numpy(idx)).cuda()

        # CPC loss = -E_X[log(f(x_t+1, z_t)/sum(f(x_i, z_t)))]
        if torch.cuda.is_available():
            x = torch.index_select(data, 0, idx)[:, :x_dim]
            x_next = torch.index_select(data, 0, idx)[:, x_dim:]
        else:
            x = data[idx, :x_dim]
            x_next = data[idx, x_dim:]
        z = C.encode(x)

        density = C.density(x_next, z)
        density_sum = 0
        #for j in range(batch_size):
        #    density_sum += C.density(torch.cat((x_next[-j:], x_next[:-j])), z)
        # filter out negative samples that are too far from the point
        density_sum += density
        for j in range(batch_size):
            negative_sample = torch.zeros(0).cuda()
            for k in range(batch_size):
                sample_idx = np.random.choice(data_size, size=1)[0]
                sample_x = data[sample_idx, :x_dim].cpu()
                true_x = data[idx[k], x_dim:].cpu()
                dist = np.sqrt((sample_x[0] - true_x[0])**2 + (sample_x[1] - true_x[1])**2)
                while sample_idx == idx[k] or dist < 0.05:
                    sample_idx = np.random.choice(data_size, size=1)[0]
                    sample_x = data[sample_idx, :x_dim].cpu()
                    true_x = data[idx[k], x_dim:].cpu()
                    dist = np.sqrt((sample_x[0] - true_x[0])**2 + (sample_x[1] - true_x[1])**2)
                if torch.cuda.is_available():
                    sample_idx = Variable(torch.from_numpy(np.array([sample_idx]))).cuda()
                    negative_sample = torch.cat((negative_sample, torch.index_select(data, 0, sample_idx)[:, :x_dim]))
                else:
                    negative_sample = torch.cat((negative_sample, data[sample_idx, :x_dim]))
            negative_sample = torch.reshape(negative_sample, (batch_size, x_dim))
            density_sum += C.density(negative_sample, z)

        C_loss = -torch.mean(torch.log(density / density_sum))

        C_loss.backward()
        C_solver.step()

        reset_grad(params)

    print("********** Epoch %i ************" % epoch)
    print(C_loss)

    log_value('C_loss', C_loss, epoch)

    if not os.path.exists('%s/var' % savepath):
        os.makedirs('%s/var' % savepath)
    torch.save(C.state_dict(), '%s/var/cpc%d' % (savepath, epoch))
