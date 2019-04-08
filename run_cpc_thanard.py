import torch
import torch.optim as optim
import numpy as np
import os.path

from matplotlib import pyplot as plt, cm, colors, gridspec
from tensorboard_logger import configure, log_value, log_histogram
from torch.autograd import Variable
from cpc import CPC
from cpc_utils import from_numpy_to_var, reset_grad, get_map, plot_data_density, plot_map, plot_clusters, binary_to_int, onehot_to_int

# Arguments
x_dim = 2
z_dim = 10
batch_size = 256
data_size = batch_size * 150
k_steps = 1
horizon = 150
mini_data_size = int(data_size / k_steps / horizon)
seed = 42
N = 100
output_type = "onehot"
assert mini_data_size * k_steps * horizon == data_size

# Configure experiment path
savepath = "filter42"
configure("%s/var_log" % savepath, flush_secs=5)

# Set seed
torch.manual_seed(seed)

# Generate no obstacle map
map2d = get_map('tunnel')

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
C = CPC(x_dim, z_dim, batch_size, output_type=output_type)
if torch.cuda.is_available():
    C.cuda()

C_solver = optim.RMSprop(list(C.parameters()), lr=1e-3)
params = list(C.parameters())

# Train CPC model
for epoch in range(1000):
    n_batch = int(data_size / batch_size)
    for it in range(n_batch):
        idx = np.random.choice(data_size, size=batch_size)
        x = data[idx, :x_dim]
        x_next = data[idx, x_dim:]
        z = C.encode(x)

        # import ipdb; ipdb.set_trace()
        # Positive
        positive_log_density = C.log_density(x_next, z)

        # Negative
        negative_idx = np.random.choice(data_size, size=batch_size*N)
        negative_x = data[negative_idx, :x_dim]
        negative_log_density = C.log_density(negative_x, z.repeat(N, 1)).reshape(batch_size, N)

        # Loss
        density_ratio = 1+torch.sum(torch.exp(negative_log_density-positive_log_density[:, None]), dim=1)
        C_loss = -torch.mean(torch.log(1/density_ratio))
        # print(C_loss)
        if C_loss != C_loss:
            import ipdb
            ipdb.set_trace()
        C_loss.backward()
        C_solver.step()

        reset_grad(params)

    # Plot grids
    if output_type in ['binary', 'onehot']:
        plot_res = 30
        xv, yv = np.meshgrid(np.linspace(-1.0, 1.0, plot_res), np.linspace(-1.0, 1.0, plot_res))
        _input = np.concatenate([np.reshape(xv, (-1, 1)), np.reshape(yv, (-1, 1))], axis=1)
        z_eval = C.encode(from_numpy_to_var(_input)).data.cpu().numpy()
        if output_type == "binary":
            idx_eval = binary_to_int(z_eval, width=z_dim)
        else:
            idx_eval = onehot_to_int(z_eval)
        # import ipdb; ipdb.set_trace()
        idx_map = np.reshape(idx_eval, (plot_res, plot_res))
        plot_clusters(idx_map, z_dim, map2d)
        if epoch % 1 == 0:
            plt.savefig("%s/%03d" % (savepath, epoch))

    print("********** Epoch %i ************" % epoch)
    print(C_loss)

    log_value('C_loss', C_loss, epoch)

    if not os.path.exists('%s/var' % savepath):
        os.makedirs('%s/var' % savepath)
    torch.save(C.state_dict(), '%s/var/cpc%d' % (savepath, epoch))
