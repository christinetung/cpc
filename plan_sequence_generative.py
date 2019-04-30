import numpy as np
import os.path
import torch
import networkx as nx

from cpcvae import CPCVAE
from vae import VAE
from matplotlib import pyplot as plt
from torch.autograd import Variable

# Arguments
z_dim = 8
seed = 0

# Set seed
torch.manual_seed(seed)
np.random.seed(seed)

# Load CPC-VAE model
C = CPCVAE()
C.load_state_dict(torch.load('cpcvae', map_location='cpu'))

# Load data
'''data_file = 'data.npy'
data = np.load(data_file)
n_trajs = len(data)
print('Number of trajectories: %d' % n_trajs) #315
print('Number of transitions: %d' % sum([len(data[i]) for i in range(n_trajs)])) #19215'''

'''transitions = []
for i in range(10):
    traj = data[i]
    for t in range(0, len(traj)):
        transitions.append(traj[t])
transitions = np.array(transitions)
print('Number of transitions: %d' % len(transitions)) #610'''

# Find positive average
'''positive_val = 0
num_positive = 0
for traj in data:
    for t in range(1, len(traj)):
        o = traj[t - 1]
        o_next = traj[t]
        x = Variable(torch.FloatTensor(np.transpose(o, (2, 0, 1))[None]))
        x_next = Variable(torch.FloatTensor(np.transpose(o_next, (2, 0, 1))[None]))
        z, _ = C.encode(x)
        score = C.density(x_next, z)
        positive_val += score
        num_positive += 1
positive_avg = positive_val / num_positive
print(positive_val) #125672.9219
print(num_positive) #18900
print(positive_avg) #6.6494'''
positive_avg = 6.6494

# Find negative average
'''negative_val = 0
num_negative = 0
for i in range(n_trajs):
    for t in range(min(len(data[i]), 10)):
        o = data[i][t]
        x = Variable(torch.FloatTensor(np.transpose(o, (2, 0, 1))[None]))
        z, _ = C.encode(x)
        other_trajs = np.random.choice(n_trajs, size=15)
        for j in other_trajs:
            if i != j:
                r = np.random.randint(len(data[j]))
                o_other = data[j][r]
                x_other = Variable(torch.FloatTensor(np.transpose(o_other, (2, 0, 1))[None]))
                score = C.density(x_other, z)
                negative_val += score
                num_negative += 1
negative_avg = negative_val / num_negative
print(negative_val) #2060.9458
print(num_negative) #47096
print(negative_avg) #0.0438'''
negative_avg = 0.0438

# Calculate threshold
threshold = positive_avg * 0.75 + negative_avg * 0.25
print(threshold) #4.9980 75-20

# Sample generated images
'''device = torch.device('cpu')
V = VAE(z_dim=64).to(device)
V.load_state_dict(torch.load('z_64_seed_0_beta_5/var/vae', map_location='cpu'))
generated = []
for i in range(8):
    sample_z = torch.randn(64, 64).to(device)
    sample_o = V.decode(sample_z).cpu()
    for j in range(64):
        sample = np.transpose(sample_o.data.cpu().numpy()[j], (1, 2, 0))
        generated.append(sample)
generated = np.array(generated)
np.save('generated.npy', generated)'''
'''generated = np.load('generated.npy')
for i in range(len(generated)):
    plt.imshow(generated[i])
    plt.savefig('images/%d.png' % i)'''

# Make graph
G = nx.Graph()
G.add_nodes_from(range(len(generated)))
for i in range(len(generated)):
    o = generated[i]
    if torch.cuda.is_available():
        x = Variable(torch.cuda.FloatTensor(np.transpose(o, (2, 0, 1))[None]))
    else:
        x = Variable(torch.FloatTensor(np.transpose(o, (2, 0, 1))[None]))
    z, _ = C.encode(x)
    tscores = torch.zeros(len(generated))
    for j in [k for k in range(len(generated)) if k != i]:
        o_other = generated[j]
        if torch.cuda.is_available():
            x_other = Variable(torch.cuda.FloatTensor(np.transpose(o_other, (2, 0, 1))[None]))
        else:
            x_other = Variable(torch.FloatTensor(np.transpose(o_other, (2, 0, 1))[None]))
        score = C.density(x_other, z)
        if score > threshold:
            tscores[j] = score
    topt = torch.argsort(tscores, descending=True)[:5]
    for t in topt:
        if tscores[t] != 0:
            G.add_edge(i, t.data.cpu().numpy().item(), weight=tscores[t].data.cpu().numpy().item())
            print('%d - %d - %f' % (i, t, tscores[t]))
nx.write_gml(G, 'graph.gml')
# Find path sequence
start = 6
end = 4
path = nx.astar_path(G, start, end)
print(path)
for p in range(len(path)):
    plt.imshow(generated[path[p]])
    plt.savefig('sequence-%d-%d/%d-%d.png' % (start, end, p, path[p]))

'''generated = np.load('generated.npy')
G = nx.read_gml('graph.gml')
path = nx.astar_path(G, '15', '12')
print(path)
for p in range(len(path)):
    plt.imshow(generated[int(path[p])])
    plt.savefig('sequence-15-12/%d-%d.png' % (p, int(path[p])))'''
