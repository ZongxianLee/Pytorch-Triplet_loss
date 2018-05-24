from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import csv

class c3d_datasets(Dataset):
    def __init__(self, pair, feature, flag = 'train', transform = None):
        """
        Args: 
            csv_files: the sample and negative/positive and its label(from csv_file)
            feature_path: a numpy numdarry and the feature vector is available by indexing
        """
        self.pair = pair
        self.feature = feature
        self.flag = flag
        self.transform = transform
    def __getitem__(self, index):
        """
        concat the features and decide wether it is relevance
        """
        
        data_1 = self.feature[int(self.pair[index][0])]
        data_1.resize((1, 512))
        
        data_2 = self.feature[int(self.pair[index][1])]
        data_2.resize((1, 512))
        
        data_3 = self.feature[int(self.pair[index][2])]
        data_3.resize((1, 512))
        
        sample = {'data_1': data_1, 'data_2': data_2, 'data_3': data_3}
        
        return sample
    def __len__(self):
        return len(self.pair)
