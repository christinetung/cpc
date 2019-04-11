import numpy as np
import os.path
import torch
import torch.optim as optim
import matplotlib

from blockwall.cpc import CPC
from matplotlib import pyplot as plt
from tensorboard_logger import configure, log_value
from torch.autograd import Variable
from blockwall.utils import reset_grad, from_numpy_to_var, binary_to_int, onehot_to_int
from blockwall.dataset import ImageNumpy
from torch.utils.data import DataLoader

# Arguments
z_dim = 10
k_steps = 5
num_epochs = 1000
batch_size = 8
N = 50
seed = 0
output_type = "binary"
c_dim = 2**z_dim if output_type == "binary" else z_dim
eval_size = 400
# Configure experiment path
savepath = os.path.join('out',
                        'blockwall',
                        output_type,
                        'z_%d_seed_%d' % (z_dim, seed))
configure('%s/var_log' % savepath, flush_secs=5)

# Set seed
torch.manual_seed(seed)
np.random.seed(seed)

# Load data
data_file = 'blockwall/obst_data2.npy'
data = np.load(data_file)
n_trajs = len(data)
data_size = sum([len(data[i])-k_steps for i in range(n_trajs)])
print('Number of trajectories: %d' % n_trajs) #315
print('Number of transitions: %d' % data_size) #378315

'''for i in range(0, len(data[0]), k_steps):
    item = data[0]
    img = item[i][0] #(64, 64, 3)
    plt.imshow(img)
    plt.savefig('%d.png' % i)
    state = item[i][1]['state']
    print(state)'''
# Dataset
# dataset = ImageNumpy(root=data_file,
#                      transform=None,
#                      k_steps=k_steps)
# data_loader = DataLoader(dataset=dataset,
#                          # pin_memory=True,
#                          num_workers=1,
#                          batch_size=batch_size,
#                          shuffle=False)

# Create CPC model
C = CPC(output_type=output_type)
if torch.cuda.is_available():
    C.cuda()
C_solver = optim.RMSprop(list(C.parameters()), lr=1e-3)
params = list(C.parameters())

def get_torch_images_from_numpy(npy_list):
    """
    :param npy_list: a list of (image, attrs) pairs
    :return: Torch Variable as input to model
    """
    return from_numpy_to_var(np.transpose(np.stack(npy_list[:, 0]), (0, 3, 1, 2)))


# Preprocess data
# MEM ERROR
# import ipdb; ipdb.set_trace()
# data = {"o": get_torch_images_from_numpy(data[:, :-k_steps].reshape(-1, 2)),
#         "o_next": get_torch_images_from_numpy(data[:, k_steps:]).reshape(-1, 2),
#         "np_pos": np.stack([_['state'][1,:2] for _ in data[:, :-k_steps].reshape(-1, 2)[:, 1]])
#         }

def get_idx_t(batch_size):
    idx = np.random.choice(n_trajs, size=batch_size)
    # Safe but slower --- doesn't assume the same length across trajectories
    t = np.array([np.random.randint(len(data[i]) - k_steps) for i in idx])
    return idx, t

# Train model
for epoch in range(num_epochs):
    # for iter, (img1, img2) in enumerate(data_loader):
    #     import ipdb
    #     ipdb.set_trace()
    #     pass
    n_batch = int(data_size / batch_size/100) # Reduce the number of data in one epoch by 100
    for it in range(n_batch):
        # idx = np.random.choice(n_trajs, size=batch_size)
        # t = np.array([np.random.randint(len(data[i]) - k_steps) for i in idx])
        idx, t = get_idx_t(batch_size)
        o = get_torch_images_from_numpy(data[idx, t])
        o_next = get_torch_images_from_numpy(data[idx, t + k_steps])

        # idx = np.random.choice(data_size, size=batch_size)
        # o = data["o"][idx]
        # o_next = data["o_next"][idx]
        z = C.encode(o)

        # Positive
        positive_log_density = C.log_density(o_next, z)

        # Negative
        negative_idx, negative_t = get_idx_t(batch_size*N)
        negative_o = get_torch_images_from_numpy(data[negative_idx, negative_t])
        negative_log_density = C.log_density(negative_o, z.repeat(N, 1)).reshape(batch_size, N)

        # Loss
        density_ratio = 1+torch.sum(torch.exp(negative_log_density-positive_log_density[:, None]), dim=1)
        C_loss = -torch.mean(torch.log(1/density_ratio))

        if epoch == 0:
            break

        # Training
        C_loss.backward()
        C_solver.step()

        reset_grad(params)

        # import ipdb
        # ipdb.set_trace()
        # if torch.cuda.is_available():
        #     x = Variable(torch.cuda.FloatTensor(np.transpose(o, (2, 0, 1))[None]))
        #     x_next = Variable(torch.cuda.FloatTensor(np.transpose(o_next, (2, 0, 1))[None]))
        # else:
        #     x = Variable(torch.FloatTensor(np.transpose(o, (2, 0, 1))[None]))
        #     x_next = Variable(torch.FloatTensor(np.transpose(o_next, (2, 0, 1))[None]))
        # z = C.encode(x)
        # density_x = C.density(x_next, z)
        # density_sum = 0
        # for j in [n for n in range(n_trajs) if n != i]:
        #     k = np.random.randint(len(data[j]))
        #     o_other = data[j][k][0]
        #     if torch.cuda.is_available():
        #         x_other = Variable(torch.cuda.FloatTensor(np.transpose(o_other, (2, 0, 1))[None]))
        #     else:
        #         x_other = Variable(torch.FloatTensor(np.transpose(o_other, (2, 0, 1))[None]))
        #     density_sum += torch.exp(C.density(x_other, z) - density_x)
        # density = 1.0 / (1.0 + density_sum)
        # C_loss = -torch.mean(torch.log(density))
        # C_loss.backward()
        # C_solver.step()
        # reset_grad(params)

    print('********** Epoch %i ************' % epoch)
    print(C_loss.cpu().data)
    log_value('C_loss', C_loss, epoch)

    if not os.path.exists('%s/var' % savepath):
        os.makedirs('%s/var' % savepath)
    torch.save(C.state_dict(), '%s/var/cpc-%d-last-5' % (savepath, (epoch-1) % 5 + 1))

    # Plot
    if output_type in ['binary', 'onehot']:
        idx, t = get_idx_t(eval_size)
        o = get_torch_images_from_numpy(data[idx, t])
        z = C.encode(o)
        if output_type == 'binary':
            y = binary_to_int(z.detach().cpu().numpy())
        else:
            y = onehot_to_int(z.detach().cpu().numpy())
        np_pos_o = np.stack([_['state'][1,:2] for _ in data[idx, t][:, 1]])
        cmap = matplotlib.cm.get_cmap('hsv')
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=c_dim)
        fig = plt.figure()
        plt.scatter(np_pos_o[:, 0], np_pos_o[:, 1], c=y,  cmap=cmap, norm=norm)
        if epoch % 1 == 0:
            # import ipdb; ipdb.set_trace()
            plt.savefig("%s/%03d" % (savepath, epoch))
    # if output_type in ['binary', 'onehot']:
    #     plot_res = 31
    #     xv, yv = np.meshgrid(np.linspace(-1.0, 1.0, plot_res), np.linspace(-1.0, 1.0, plot_res))
    #     _input = np.concatenate([np.reshape(xv, (-1, 1)), np.reshape(yv, (-1, 1))], axis=1)
    #     z_eval = C.encode(from_numpy_to_var(_input)).data.cpu().numpy()
    #     if output_type == "binary":
    #         idx_eval = binary_to_int(z_eval, width=z_dim)
    #         n_colors = 2**z_dim
    #     else:
    #         idx_eval = onehot_to_int(z_eval)
    #         n_colors = z_dim
    #     # import ipdb; ipdb.set_trace()
    #     idx_map = np.reshape(idx_eval, (plot_res, plot_res))
    #     plot_clusters(idx_map, n_colors, map2d)
    #     if epoch % 1 == 0:
    #         plt.savefig("%s/%03d" % (savepath, epoch))
