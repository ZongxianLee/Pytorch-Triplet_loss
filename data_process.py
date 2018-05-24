import csv
import numpy as np
import os


def get_data_pair(csv_name):
    pair_data = []
    csv_file = csv.reader(open(csv_name))
    for data in csv_file:
        pair_data.append(data)
    return pair_data
def load_feature(npy_file):
    features = np.load(npy_file)
    return features
