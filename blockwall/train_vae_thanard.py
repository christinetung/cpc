import numpy as np
import os.path
import torch
import torch.optim as optim
import matplotlib
from torchvision.utils import save_image
from blockwall.cpc import VAE, loss_function
from matplotlib import pyplot as plt
from tensorboard_logger import configure, log_value
from torch.autograd import Variable
from blockwall.utils import reset_grad, from_numpy_to_var, binary_to_int, onehot_to_int
from blockwall.dataset import ImageNumpy
from torch.utils.data import DataLoader
from mpl_toolkits.mplot3d import Axes3D

# Arguments
z_dim = 8
k_steps = 1
num_epochs = 1000
batch_size = 32
N = 50
seed = 0
output_type = "continuous"
c_dim = 2**z_dim if output_type == "binary" else z_dim
eval_size = 400
vae_weight = 0.001
# Configure experiment path
savepath = os.path.join('out',
                        'blockwall-vae',
                        output_type,
                        'z_%d_vae_%.3f_batch_size_%d_N_%d_k_%d_seed_%d' % (z_dim, vae_weight, batch_size, N, k_steps, seed))
configure('%s/var_log' % savepath, flush_secs=5)

# Set seed
torch.manual_seed(seed)
np.random.seed(seed)

colormap = np.random.random(size=(c_dim, 3))

# Load data
data_file = 'blockwall/obst_data2.npy'
data = np.load(data_file)
n_trajs = len(data)
data_size = sum([len(data[i])-k_steps for i in range(n_trajs)])
print('Number of trajectories: %d' % n_trajs) #315
print('Number of transitions: %d' % data_size) #378315

# Create CPC model
cpc = VAE(z_dim, output_type=output_type)
if torch.cuda.is_available():
    cpc.cuda()
C_solver = optim.Adam(list(cpc.parameters()), lr=1e-3)
# C_solver = optim.RMSprop(list(C.parameters()), lr=1e-3)
params = list(cpc.parameters())


def get_torch_images_from_numpy(npy_list, normalize=True):
    """
    :param npy_list: a list of (image, attrs) pairs
    :param normalize: if True then the output is between 0 and 1
    :return: Torch Variable as input to model
    """
    return from_numpy_to_var(np.transpose(np.stack(npy_list[:, 0]), (0, 3, 1, 2)))/255


def get_idx_t(batch_size):
    idx = np.random.choice(n_trajs, size=batch_size)
    # Safe but slower --- doesn't assume the same length across trajectories
    t = np.array([np.random.randint(len(data[i]) - k_steps) for i in idx])
    return idx, t


# Train model
for epoch in range(num_epochs):
    n_batch = int(data_size / batch_size/100) # Reduce the number of data in one epoch by 100
    for it in range(n_batch):
        idx, t = get_idx_t(batch_size)
        o = get_torch_images_from_numpy(data[idx, t])
        o_next = get_torch_images_from_numpy(data[idx, t + k_steps])
        o_pred, mu, logvar = cpc(o)
        o_next_pred, _, _ = cpc(o_next)
        z_pred, _, _ = cpc.encode(o_pred)

        # Positive
        positive_log_density = cpc.log_density(o_next_pred, z_pred)

        # Negative
        negative_idx, negative_t = get_idx_t(batch_size*N)
        negative_o = get_torch_images_from_numpy(data[negative_idx, negative_t])
        negative_o_pred, _, _ = cpc(negative_o)
        negative_log_density = cpc.log_density(negative_o_pred, z_pred.repeat(N, 1)).reshape(batch_size, N)

        # Loss
        density_ratio = 1+torch.sum(torch.exp(negative_log_density-positive_log_density[:, None]), dim=1)
        C_loss = -torch.mean(torch.log(1/density_ratio))
        # import ipdb
        # ipdb.set_trace()
        VAE_loss = loss_function(o_pred, o, mu, logvar)

        if epoch == 0:
            break

        # Training
        if epoch < 10:
            VAE_loss.backward()
        else:
            (C_loss + vae_weight*VAE_loss).backward()
        C_solver.step()

        reset_grad(params)

    print('********** Epoch %i ************' % epoch)
    print(C_loss.cpu().data)
    print(VAE_loss.cpu().data)
    log_value('C_loss', C_loss, epoch)
    log_value('VAE_loss', VAE_loss, epoch)

    if not os.path.exists('%s/var' % savepath):
        os.makedirs('%s/var' % savepath)
    torch.save(cpc.state_dict(), '%s/var/cpc-%d-last-5' % (savepath, epoch % 5 + 1))

    comparison = torch.cat([o[:8], o_pred[:8]])
    save_image(comparison.data.cpu(),
               os.path.join(savepath, 'reconstruction_' + str(epoch) + '.png'), nrow=8)

    # Plot
    if epoch % 1 == 0:
        # Clusters
        if output_type in ['binary', 'onehot']:
            idx, t = get_idx_t(eval_size)
            o = get_torch_images_from_numpy(data[idx, t])
            z, _, _ = cpc.encode(o)
            if output_type == 'binary':
                y = binary_to_int(z.detach().cpu().numpy())
            else:
                y = onehot_to_int(z.detach().cpu().numpy())
            print('# clusters', len(np.unique(y)))
            np_pos_o = np.stack([_['state'][1,:2] for _ in data[idx, t][:, 1]])
            # cmap = matplotlib.cm.get_cmap('hsv')
            # norm = matplotlib.colors.Normalize(vmin=0.0, vmax=c_dim)
            fig = plt.figure()
            # plt.scatter(np_pos_o[:, 0], np_pos_o[:, 1], c=y,  cmap=cmap, norm=norm)
            # import ipdb; ipdb.set_trace()
            plt.scatter(np_pos_o[:, 0], np_pos_o[:, 1], c=colormap[y])
            plt.savefig("%s/%03d" % (savepath, epoch))
            plt.close()

            # 3D
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(np_pos_o[:, 0], np_pos_o[:, 1], y, c=colormap[y])
            plt.savefig("%s/d3_%03d" % (savepath, epoch))
            plt.close()

        # Energy model

