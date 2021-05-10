# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 11:12:20 2021

@author: jwehounou
"""
import time
import pandas
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



# function to generate real data
def generate_real():
    real_data = torch.FloatTensor(
    [random.uniform(0.8, 1.0),
    random.uniform(0.0, 0.2),
    random.uniform(0.8, 1.0),
    random.uniform(0.0, 0.2)])
    return real_data

def generate_random(size):
    random_data = torch.rand(size)
    return random_data

class Discriminator(nn.Module):
    
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()
        
        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(4, 3),
            nn.Sigmoid(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
        
        # create loss function
        self.loss_function = nn.MSELoss()

        # create optimiser, using stochastic gradient descent
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

        # counter and accumulator for progress
        self.counter = 0
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
    
        # increase counter and accumulate error every 10 epochs
        self.counter += 1
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
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        pass

class Generator(nn.Module):
    
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()
        
        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(1, 3),
            nn.Sigmoid(),
            nn.Linear(3, 4),
            nn.Sigmoid()
        )

        # create optimiser, simple stochastic gradient descent
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

        # counter and accumulator for progress
        self.counter = 0
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
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
    
        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
    
        pass
    
    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        pass


def train_and_testing_discriminator(size=4):
    
    D = Discriminator()
    
    for i in range(10000):
        # real data
        D.train(generate_real(), torch.FloatTensor([1.0]))
        # fake data
        D.train(generate_random(size), torch.FloatTensor([0.0]))
        pass
    
    D.plot_progress()
    plt.savefig('output/legend.png')


    print("Real data source:", D.forward(generate_real()).item())

    print("Random noise:", D.forward(generate_random(4)).item())
    
def testing_generator():
    # creating discriminator object
    D = Discriminator()
    # creating generator object
    G = Generator()
    G.forward(torch.FloatTensor([0.5]))
    
    pass
    
def train_GAN():
    # create Discriminator and Generator
    D = Discriminator()
    G = Generator()
    
    # train Discriminator and Generator
    for i in range(10000):
        
        # train discriminator on true
        D.train(generate_real(), torch.FloatTensor([1.0]))
        
        # train discriminator on false
        # use detach() so gradients in G are not calculated
        D.train(G.forward(torch.FloatTensor([0.5])).detach(), torch.FloatTensor([0.0]))
        
        # train generator
        G.train(D, torch.FloatTensor([0.5]), torch.FloatTensor([1.0]))
    
    D.plot_progress()
    plt.xlabel('Discriminator loss chart')
    plt.savefig('output/Discriminator.png')

    G.plot_progress()
    plt.xlabel('Generator loss chart')
    plt.savefig('output/Generator.png')
    
    
    print("Output of trained generator:", G.forward(torch.FloatTensor([0.5])))
    pass

if __name__ == "__main__":
    ti = time.time()
    print("Real random value:", generate_real())