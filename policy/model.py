import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import argparse
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch as t
from torch.autograd import Variable
from policy.dataset import ImageAction
import torch.optim as optim
import os
# from utils import *
from random import shuffle as shuffle
import random


class Policy(nn.Module):
    def __init__(self, in_channels=3, action_dim=6, f=64):
        super(Policy, self).__init__()

        self.main = nn.Sequential(
            # input size (2 or 6) x 64 x64
            nn.Conv2d(2*in_channels, f, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.5),
            # 64 x 32 x 32
            nn.Conv2d(f, f*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f*2),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.5),
            # 128 x 16 x 16
            nn.Conv2d(f*2, f*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f*4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.5),
            # Option 1: 256 x 8 x 8
            nn.Conv2d(f*4, f*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(f*8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Dropout2d(p=0.5),
            # 512 x 4 x 4
            nn.Conv2d(f*8, action_dim, 4),
            # Got rid of final activation
            nn.Tanh()
            #nn.CMul(10.0)
        )

    def forward(self, x, x_next):
        input = torch.cat([x, x_next], dim=1)
        out = self.main(input).squeeze() #* 10.0
        return out


def train(**kwargs):
    # noise = kwargs['noise']
    epochs = kwargs['n_epochs']
    batch_size = kwargs['batch_size']

    lr = 1e-3
    criterion = nn.MSELoss()

    trans = [
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
    ]

    trans_comp = transforms.Compose(trans)
    dataset_train = ImageAction(root='blockwall/obst_data2.npy',
                                test=False,
                                transform=trans_comp)
    # dataset_test = ImageAction(root='blockwall/obst_data2.npy',
    #                            test=True,
    #                            transform=trans_comp)
    print("Train data number: {}".format(len(dataset_train)))
    # print("Test data number: {}".format(len(dataset_test)))

    dataloader_train = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=2,
                                                   drop_last=True)

    # dataloader_test = torch.utils.data.DataLoader(dataset_test,
    #                                               batch_size=batch_size,
    #                                               shuffle=True,
    #                                               num_workers=2,
    #                                               drop_last=True)
    in_channels = 3
    policy = Policy(in_channels=in_channels, action_dim=6)
    if torch.cuda.is_available():
        policy.cuda()

    # initialize

    def init_weights(m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    # policy.apply(init_weights)

    optimizer = optim.Adam(policy.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=.0)
    min_test_loss = 1000000.

    for epoch in range(epochs):
        # batch_data is a tuple ((img_1, img_2), action)
        # img_1 and img_2 are both shape (64, 3, 64, 64) and action is shape (64, 6)
        train_loss = 0.0
        test_loss = 0.0
        policy.train()

        for num_iters, batch_data in enumerate(dataloader_train, 0):

            optimizer.zero_grad()

            img = batch_data[0][0].float()
            img_next = batch_data[0][1].float()
            action_real = batch_data[1].float()
            if torch.cuda.is_available():
                img = img.cuda()
                img_next = img_next.cuda()
                action_real = action_real.cuda()
            # import ipdb; ipdb.set_trace()
            action_pred = policy(img, img_next)
            loss = criterion(action_pred, action_real)
            loss.backward()
            optimizer.step()

            train_loss += loss
            if num_iters % 100 ==0:
                print(loss)
        # policy.eval()
        # for num_iters, batch_data in enumerate(dataloader_test, 0):
        #     img = batch_data[0][0].float()
        #     img_next = batch_data[0][1].float()
        #     action_real = batch_data[1].float()
        #     action_pred = policy(img, img_next)
        #     loss = criterion(action_pred, action_real)
        #
        #     test_loss += loss

        train_loss /= len(dataset_train)
        # test_loss /= len(dataset_test)
        train_loss *= 6.0 * 1000.0
        # test_loss *= 6.0 * 1000.0

        if test_loss < min_test_loss:
            torch.save(policy.state_dict(), 'tunnel_policy_1_lowest.pth')
            min_test_loss = test_loss

        print("EPOCH {0}, TRAIN LOSS: {1}, TEST LOSS: {2}".format(epoch, train_loss, test_loss))

        torch.save(policy.state_dict(), 'tunnel_policy_1.pth')
        print("min test loss: {}".format(min_test_loss))
