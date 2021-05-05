# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:44:24 2021

@author: jwehounou
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
import torch
import torch.nn as nn
import pandas as pd
import random

import matplotlib.pyplot as plt

from Dataset import FMnistDataset

from pathlib import Path


###############################################################################
#                   discriminator class : debut
###############################################################################
class Discriminator(nn.Module):
    
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()
        
        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 1),
            nn.Sigmoid()
        )
        
        # create loss function
        self.loss_function = nn.BCELoss()

        # create optimiser, simple stochastic gradient descent
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

        # counter and accumulator for progress
        self.counter = 0;
        self.progress = []

        pass
    
    
    def forward(self, inputs):
        # simply run model
        return self.model(inputs)
    
    
    def train(self, inputs, targets):
        # calculate the output of the network
        outputs = self.forward(inputs)
        
        # calculate loss
        loss = self.loss_function(outputs, targets)

        # increase counter and accumulate error every 10
        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)
            pass

        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass
    
    
    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0), figsize=(16,8), alpha=0.1, 
                marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
        pass
  
###############################################################################
#                   discriminator class : fin
###############################################################################

###############################################################################
#               testing class and functions : debut
###############################################################################
def generate_random(size):
    """
    function to generate uniform random data    

    """
    random_data = torch.rand(size)
    return random_data

def testing_train_discriminator(mnist_dataset):
    """
    test discriminator can separate real data from random noise
    """
    
    D = Discriminator()
    
    for label, image_data_tensor, target_tensor in mnist_dataset:
        # real data
        D.train(image_data_tensor, torch.FloatTensor([1.0]))
        # fake data
        D.train(generate_random(784), torch.FloatTensor([0.0]))
        pass
    
    D.plot_progress()
    
    #plt.savefig('output/mnistDiscrimatorlegend.png')
    
    legendDiscrimator = "mnistDiscrimatorlegend.png"
    path_to_save = "output"
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    legend_discrimator_file = os.path.join(path_to_save, legendDiscrimator)
    plt.savefig(legend_discrimator_file)
    return D

def manually_randomTest_discriminator(D, mnist_dataset):
    """
    manually run discriminator to check it can tell real data from fake
    """

    for i in range(4):
      image_data_tensor = mnist_dataset[random.randint(0,60000)][1]
      print("RealData: " + D.forward( image_data_tensor ).item() )
      pass
    
    for i in range(4):
      print("FakeData: " + D.forward( generate_random(784) ).item() )
      pass
  
###############################################################################
#               testing class and functions : fin
###############################################################################

if __name__ == "__main__":
    ti = time.time()
    
    data_path = "data"
    mnist_root_dir = "mnist_data"
    train_data_name = "mnist_train.csv"
    test_data_name = "mnist_test.csv"
    csv_file = os.path.join(data_path, mnist_root_dir, train_data_name)    
    mnist_dataset = FMnistDataset(csv_file)
    
    D = testing_train_discriminator(mnist_dataset)
    #manually_randomTest_discriminator(D, mnist_dataset)
    
    print("runtime = {}".format(time.time() - ti))
    