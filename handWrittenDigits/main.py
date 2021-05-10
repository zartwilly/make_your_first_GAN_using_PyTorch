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

def learn_discriminator_generator(size=100, epochs=4):
    # create Discriminator and Generator
    D = Discriminator()
    G = Generator()
    
    for epoch in range(epochs):
        print ("epoch = ", epoch + 1, " fmnist_dataset= {}".format(len(fmnist_dataset)))
        # train Discriminator and Generator
        for label, image_data_tensor, target_tensor in fmnist_dataset:
            # train discriminator on true
            D.train(image_data_tensor, torch.FloatTensor([1.0]))
                
            # train discriminator on false
            # use detach() so gradients in G are not calculated
            D.train(G.forward(generate_random_seed(size=size)).detach(), torch.FloatTensor([0.0]))
                
            # train generator
            G.train(D, generate_random_seed(size), torch.FloatTensor([1.0]))
            pass
        pass
    
    return D, G
    
def show_created_images(G, size):
    # plot several outputs from the trained generator
    # plot a 3 column, 2 row array of generated images
    f, axarr = plt.subplots(2,3, figsize=(16,8))
    for i in range(2):
        for j in range(3):
            output = G.forward(generate_random_seed(size=size))
            img = output.detach().numpy().reshape(28,28)
            axarr[i,j].imshow(img, interpolation='none', cmap='Blues')
            pass
        pass
    
    generated_images = "generated_images_generator.png"
    path_to_save = "output"
    Path(path_to_save).mkdir(parents=True, exist_ok=True)
    generated_images_file = os.path.join(path_to_save, generated_images)
    plt.savefig(generated_images_file)


def generate_image_by_using_seeds_approach1(G, size=100, j_cols=4, i_rows=3):
    """
    approach 1: variation of the seed between 2 intervals
    """
    seed1 = generate_random_seed(size);
    seed2 = generate_random_seed(size);
    
    count = 0

    # plot a i_cols column, j_rows row array of generated images
    f, axarr = plt.subplots(i_rows,j_cols, figsize=(16,8))
    for i in range(i_rows):
        for j in range(j_cols):
            seed = seed1 + (seed2 - seed1)/(i_rows*j_cols - 1) * count
            output = G.forward(seed)
            img = output.detach().numpy().reshape(28,28)
            axarr[i,j].imshow(img, interpolation='none', cmap='Blues')
            count = count + 1
            pass
        pass
    
def generate_image_by_using_seeds_approach2(G, size=100, j_cols=4, i_rows=3):
    """
    approach 2: summation of the seeds
    """
    seed1 = generate_random_seed(size);
    seed2 = generate_random_seed(size);
    
    seed3 = seed1 + seed2
    out3 = G.forward(seed3)
    img3 = out3.detach().numpy().reshape(28,28)
    plt.imshow(img3, interpolation='none', cmap='Blues')
        


if __name__ == "__main__":
    ti = time.time()
    size = 100
    epochs = 4
    D, G = None, None

    #if D is not None and G is not None:
    D, G = learn_discriminator_generator(size=size, epochs=epochs)
    show_created_images(G=G, size=size)
    
    j_cols=4; i_rows=3
    generate_image_by_using_seeds_approach1(G, size=size, j_cols=j_cols, i_rows=i_rows)

    print("runtime = {}".format(time.time() - ti))