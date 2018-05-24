from __future__ import print_function, division
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import io
import copy
import torch.optim as optim
import time

def train(dataloaders, net, criterion, optimizer, scheduler, show_data_size, use_gpu, num_epoches = 100):
    for epoch in range(num_epoches):
        print('Epoch {}/{}'.format(epoch, num_epoches-1))
        print('-'* 20)

        
        for phase in ['train']:
            if phase == 'train':
                scheduler.step()
                net.train()
            running_loss = 0.0
           #running_positive_dis = 0.0
           #running_negative_dis = 0.0
            
            for datasets in dataloaders[phase]:
                input_1 = datasets['data_1'].float()
                input_2 = datasets['data_2'].float()
                input_3 = datasets['data_3'].float()
                optimizer.zero_grad()
                
                if use_gpu:
                    input_1 = Variable(input_1.cuda())
                    input_2 = Variable(input_2.cuda())
                    input_3 = Variable(input_3.cuda())
                
                dis_1, dis_2, embedded_x, embedded_y, embedded_z = net(input_1, input_2, input_3)
                target = torch.FloatTensor(dis_1.size()).fill_(1)
                target = Variable(target.cuda())
                
                loss_triplet = criterion(dis_1, dis_2, target)
                loss_embeded = embedded_x.norm(2) + embedded_y.norm(2) + embedded_z.norm(2)
                loss = loss_triplet + 0.001 * loss_embeded
                
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.data[0] * input_1.size(0)
                #running_positive_dis += dis_1 * input_1.size(0)
                #running_negative_dis += dis_2 * input_1.size(0)
                
            #epoch_dis_1 = running_positive_dis / show_data_size[phase]
            #epoch_dis_2 = running_negative_dis / show_data_size[phase]
            epoch_loss = running_loss / show_data_size[phase]
            #print('positive distance: {:.4f}, negative distance {:.4f}'.format(epoch_dis_1, epoch_dis_2))
            print('loss:{}'.format(epoch_loss))
            print("\n")
        
        torch.save(net.state_dict(), 'param.pkl')
            
