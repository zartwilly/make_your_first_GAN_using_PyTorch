# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:28:26 2021

@author: jwehounou
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
import torch
import matplotlib.pyplot as plt
from Dataset import FMnistDataset
from Discriminator import Discriminator
from Generator import Generator

from pathlib import Path


ti = time.time()


fmnist_dataset = None

bool_fashion = True #False
if not bool_fashion:  
    # fashion mnist load data
    data_path = "data"
    train_data_name = "fashion-mnist_train.csv"
    mnist_root_dir = "fashion_mnist_data"
    fashion_csv_file = os.path.join(data_path, mnist_root_dir, train_data_name)
    fmnist_dataset = FMnistDataset(fashion_csv_file)
else:
    # mnist load data
    data_path = "data"
    mnist_root_dir = "mnist_data"
    train_data_name = "mnist_train.csv"
    test_data_name = "mnist_test.csv"
    csv_file = os.path.join(data_path, mnist_root_dir, train_data_name)
    fmnist_dataset = FMnistDataset(csv_file)



# functions to generate random data
def generate_random_image(size):
    random_data = torch.rand(size)
    return random_data


def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data

# create Discriminator and Generator

D = Discriminator()
G = Generator()

epochs = 4

for epoch in range(epochs):
    print ("epoch = ", epoch + 1, " fmnist_dataset= {}".format(len(fmnist_dataset)))
    # train Discriminator and Generator
    for label, image_data_tensor, target_tensor in fmnist_dataset:
        # train discriminator on true
        D.train(image_data_tensor, torch.FloatTensor([1.0]))
            
        # train discriminator on false
        # use detach() so gradients in G are not calculated
        D.train(G.forward(generate_random_seed(100)).detach(), torch.FloatTensor([0.0]))
            
        # train generator
        G.train(D, generate_random_seed(100), torch.FloatTensor([1.0]))
        pass
    pass

# plot several outputs from the trained generator
# plot a 3 column, 2 row array of generated images
f, axarr = plt.subplots(2,3, figsize=(16,8))
for i in range(2):
    for j in range(3):
        output = G.forward(generate_random_seed(100))
        img = output.detach().numpy().reshape(28,28)
        axarr[i,j].imshow(img, interpolation='none', cmap='Blues')
        pass
    pass

generated_images = "generated_images_generator.png"
path_to_save = "output"
Path(path_to_save).mkdir(parents=True, exist_ok=True)
generated_images_file = os.path.join(path_to_save, generated_images)
plt.savefig(generated_images_file)

print("runtime = {}".format(time.time() - ti))