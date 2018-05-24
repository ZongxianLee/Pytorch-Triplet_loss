from __future__ import print_function, division
import torch
import numpy as np
import copy
import csv
import data_process
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from my_net import embedding_net, triplet_net
from torch.optim import lr_scheduler
import torch.nn.functional as F
from my_dataset import c3d_datasets
from train_net import train
import torch.optim as optim

current_path = os.getcwd()

# load your own csv or other format data here
train_file = os.path.join(current_path, 'xxx.csv')

# load the feature npy to the memory if possible(for a faster feature reading speed)
feature_npy = 'xxx.npy'

# data loading
train_pair = data_process.get_data_pair(train_file)

# pre-loding the feature to the memory
c3d_feature = data_process.load_feature(feature_npy)

pair = {'train': train_pair}
show_datasets = {x: c3d_datasets(pair[x], c3d_feature, x) for x in ['train']}
show_data_sizes = {x: len(show_datasets[x]) for x in ['train']}

dataloaders = {x: DataLoader(show_datasets[x], batch_size = 256, shuffle = True, num_workers = 4)
              for x in ['train']}
use_gpu = torch.cuda.is_available()


# initilize the network and the optimizer stretigies
model = embedding_net()
net = triplet_net(model)
net = net.cuda()
criterion = torch.nn.MarginRankingLoss(margin = 0.5)
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum = 0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)

train(dataloaders, net, criterion, optimizer, exp_lr_scheduler, show_data_sizes, use_gpu,  num_epoches = 100)
