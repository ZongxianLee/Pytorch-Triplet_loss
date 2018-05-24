import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class embedding_net(nn.Module):
    def __init__(self):
        super(embedding_net, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 64, 7, padding = 3)
        self.bn_conv1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, 7, padding = 3)
        self.bn_conv2 = nn.BatchNorm1d(128)
        
        self.pool = nn.MaxPool1d(1, 2)
        
        self.conv_fc = nn.Linear(128 * 256, 1024)
        self.bn_conv_fc = nn.BatchNorm1d(1024)
    
    def forward(self,inputs):
        x = self.bn_conv1(F.relu(self.conv1(inputs)))
        x = self.bn_conv2(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = x.view(-1, 128 * 256)
        x = self.bn_conv_fc(F.relu(self.conv_fc(x)))
        
        return x

    
class triplet_net(nn.Module):
    def __init__(self, embedding_net):
        super(triplet_net, self).__init__()
        self.embedding = embedding_net
    
    def forward(self, x, y, z):
        embedded_x = self.embedding(x)
        embedded_y = self.embedding(y)
        embedded_z = self.embedding(z)
        
        dis_1 = F.pairwise_distance(embedded_x, embedded_y, 2)
        dis_2 = F.pairwise_distance(embedded_x, embedded_z, 2)
        
        return dis_1, dis_2, embedded_x, embedded_y, embedded_z
