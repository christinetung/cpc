import numpy as np
import os.path
import torch
import torch.optim as optim

from cpc import CPC
from matplotlib import pyplot as plt
from tensorboard_logger import configure, log_value
from torch.autograd import Variable
from utils import reset_grad

# Arguments
z_dim = 10
k_steps = 10
num_epochs = 1000
batch_size = 128
seed = 0

# Configure experiment path
savepath = 'z_%d_seed_%d' % (z_dim, seed)
configure('%s/var_log' % savepath, flush_secs=5)

# Set seed
torch.manual_seed(seed)
np.random.seed(seed)

# Load data
data_file = 'obst_data2.npy'
data = np.load(data_file)
n_trajs = len(data)
print('Number of trajectories: %d' % n_trajs) #315
print('Number of transitions: %d' % sum([len(data[i]) for i in range(n_trajs)])) #378315

'''for i in range(0, len(data[0]), k_steps):
    item = data[0]
    img = item[i][0] #(64, 64, 3)
    plt.imshow(img)
    plt.savefig('%d.png' % i)
    state = item[i][1]['state']
    print(state)'''

# Create CPC model
C = CPC()
if torch.cuda.is_available():
    C.cuda()
C_solver = optim.RMSprop(list(C.parameters()), lr=1e-3)
params = list(C.parameters())

# Train model
for epoch in range(num_epochs):
    idx = np.random.choice(n_trajs, size=batch_size)
    if torch.cuda.is_available():
        idx = Variable(torch.from_numpy(idx)).cuda()
    for i in idx:
        t = np.random.randint(len(data[i]) - 1)
        o = data[i][t][0]
        o_next = data[i][t + 1][0]
        if torch.cuda.is_available():
            x = Variable(torch.cuda.FloatTensor(np.transpose(o, (2, 0, 1))[None]))
            x_next = Variable(torch.cuda.FloatTensor(np.transpose(o_next, (2, 0, 1))[None]))
        else:
            x = Variable(torch.FloatTensor(np.transpose(o, (2, 0, 1))[None]))
            x_next = Variable(torch.FloatTensor(np.transpose(o_next, (2, 0, 1))[None]))
        z = C.encode(x)
        density_x = C.density(x_next, z)
        density_sum = 0
        for j in [n for n in range(n_trajs) if n != i]:
            k = np.random.randint(len(data[j]))
            o_other = data[j][k][0]
            if torch.cuda.is_available():
                x_other = Variable(torch.cuda.FloatTensor(np.transpose(o_other, (2, 0, 1))[None]))
            else:
                x_other = Variable(torch.FloatTensor(np.transpose(o_other, (2, 0, 1))[None]))
            density_sum += torch.exp(C.density(x_other, z) - density_x)
        density = 1.0 / (1.0 + density_sum)
        C_loss = -torch.mean(torch.log(density))
        C_loss.backward()
        C_solver.step()
        reset_grad(params)

    print('********** Epoch %i ************' % epoch)
    print(C_loss)
    log_value('C_loss', C_loss, epoch)

    if not os.path.exists('%s/var' % savepath):
        os.makedirs('%s/var' % savepath)
    torch.save(C.state_dict(), '%s/var/cpc%d' % (savepath, epoch))
