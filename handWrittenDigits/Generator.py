# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:01:29 2021

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
from Discriminator import Discriminator

from pathlib import Path


###############################################################################
#                   generator class : debut
###############################################################################
class Generator(nn.Module):
    
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()
        
        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(100, 200),
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 784),
            nn.Sigmoid()
        )
        
        # create optimiser, simple stochastic gradient descent
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

        # counter and accumulator for progress
        self.counter = 0;
        self.progress = []
        
        pass
    
    
    def forward(self, inputs):        
        # simply run model
        return self.model(inputs)
    
    
    def train(self, D, inputs, targets):
        # calculate the output of the network
        g_output = self.forward(inputs)
        
        # pass onto Discriminator
        d_output = D.forward(g_output)
        
        # calculate error
        loss = D.loss_function(d_output, targets)

        # increase counter and accumulate error every 10
        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass

        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass
    
    
    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', 
                grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
        pass
    
    pass

###############################################################################
#                   generator class : fin
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

def generate_random_seed(size):
    """
    function to generate normal random data    
    """
    random_data = torch.randn(size)
    return random_data

def test_output_generator():
    # check the generator output is of the right type and shape
    G = Generator()
    output = G.forward(generate_random(100))
    img = output.detach().numpy().reshape(28,28)
    plt.imshow(img, interpolation='none', cmap='Blues')
    
    # difference between 2 outputs' generators
    output_1 = G.forward(generate_random(100))
    output_2 = G.forward(generate_random(100))
    diff = output_2 - output_1
    img = diff.detach().numpy().reshape(28,28)
    plt.imshow(img, interpolation='none', cmap='Blues')
    
def train_mnistGAN():
    # create Discriminator and Generator
    D = Discriminator()
    G = Generator()
    
    # train Discriminator and Generator    
    for label, image_data_tensor, target_tensor in mnist_dataset:
        
        # train discriminator on true
        D.train(image_data_tensor, torch.FloatTensor([1.0]))
        
        # train discriminator on false
        # use detach() so gradients in G are not calculated
        D.train(G.forward(generate_random(100)).detach(), torch.FloatTensor([0.0]))
        
        # train generator
        G.train(D, generate_random(100), torch.FloatTensor([1.0]))
    
        pass
    
    # plot GAN Discriminator error
    D.plot_progress()
    
    # plot GAN Generator errot
    G.plot_progress()
    
    return D, G

def improved_train_mnistGAN():
    # create Discriminator and Generator
    D = Discriminator()
    G = Generator()
        
    epochs = 4
    
    for epoch in range(epochs):
      print ("epoch = ", epoch + 1)
    
      # train Discriminator and Generator
    
      for label, image_data_tensor, target_tensor in mnist_dataset:
        # train discriminator on true
        D.train(image_data_tensor, torch.FloatTensor([1.0]))
        
        # train discriminator on false
        # use detach() so gradients in G are not calculated
        D.train(G.forward(generate_random_seed(100)).detach(), torch.FloatTensor([0.0]))
        
        # train generator
        G.train(D, generate_random_seed(100), torch.FloatTensor([1.0]))
    
        pass
        
      pass
  
    # plot GAN Discriminator error
    D.plot_progress()
    
    # plot GAN Generator errot
    G.plot_progress()
    
    return D, G
    
def plot_some_images_from_trained_generator(G):
    """
    plot several outputs from the trained generator
    
    """
    # plot a 3 column, 2 row array of generated images
    f, axarr = plt.subplots(2,3, figsize=(16,8))
    for i in range(2):
        for j in range(3):
            output = G.forward(generate_random(100))
            img = output.detach().numpy().reshape(28,28)
            axarr[i,j].imshow(img, interpolation='none', cmap='Blues')
            pass
        pass
    
    legendGenerator = "mnistGeneratorlegend.png"
    path_to_save = "output"
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    legend_generator_file = os.path.join(path_to_save, legendGenerator)
    plt.savefig(legend_generator_file)
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
    
    # test output generator
    test_output_generator()
    
    # train GAN
    D, G = train_mnistGAN()
    
    # plot several images from G to understand how could G learn 
    plot_some_images_from_trained_generator(G)
    
    print("runtime = {}".format(time.time() - ti))