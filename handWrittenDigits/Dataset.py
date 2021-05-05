# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:14:32 2021

@author: jwehounou
"""
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pandas, numpy, random
import matplotlib.pyplot as plt

# dataset class
class FMnistDataset(Dataset):
    
    def __init__(self, csv_file):
        self.data_df = pandas.read_csv(csv_file, header=0)
        pass
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        # image target (label)
        label = self.data_df.iloc[index,0]
        target = torch.zeros((10))
        target[label] = 1.0
        
        # image data, normalised from 0-255 to 0-1
        image_values = torch.FloatTensor(self.data_df.iloc[index,1:].values) / 255.0
        
        # return label, image data tensor and target tensor
        return label, image_values, target
    
    def plot_image(self, index):
        img = self.data_df.iloc[index,1:].values.reshape(28,28)
        plt.title("label = " + str(self.data_df.iloc[index,0]))
        plt.imshow(img, interpolation='none', cmap='Blues')
        pass
    
    pass


if __name__ == "__main__":
    data_path = "data"
    mnist_root_dir = "mnist_data"
    train_data_name = "mnist_train.csv"
    test_data_name = "mnist_test.csv"
    
    csv_file = os.path.join(data_path, mnist_root_dir, train_data_name)
    
    data = FMnistDataset(csv_file)
    
    # plot the nineth image
    data.plot_image(9)
    
    # fashion mnist data
    train_data_name = "fashion-mnist_train.csv"
    mnist_root_dir = "fashion_mnist_data"
    fashion_csv_file = os.path.join(data_path, mnist_root_dir, train_data_name)
    fmnist_dataset = FMnistDataset(fashion_csv_file)
    fmnist_dataset.plot_image(9)
    
