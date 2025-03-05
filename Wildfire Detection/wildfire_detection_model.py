# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 13:59:51 2025

@author: Veron
"""

import os
# Import necessary functions from python files
# Model, Custom dataset and data extraction function
from Data_and_Model import read_text_files_with_labels, CustomDataset, GarbageModel

# Metrics
from Metrics import metrics_eval


#---------- Importing useful packages --------------#
import torch # pytorch main library
import glob
import torchvision # computer vision utilities
import torchvision.transforms as transforms # transforms used in the pre-processing of the data
from torchvision import *

from PIL import Image
from torchvision.models import resnet18, resnet50, ResNet50_Weights
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

#import hiddenlayer as hl
#from torchviz import make_dot

import time
import copy
import re

# from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
# import seaborn as sns


# Check if GPU is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
##-----------------------------------------------------------------------------------------------------------##
##-----------------------------------------------------------------------------------------------------------##
# set paths to retrieve data
TRAIN_PATH = r"/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Train"
VAL_PATH = r"/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Val"
TEST_PATH = r"/work/TALC/enel645_2024f/garbage_data/CVPR_2024_dataset_Test"

##-----------------------------------------------------------------------------------------------------------##
# Set the hyperparameters
batch_size = 256 # Change Batch Size o
learning_rate = 1e-4 #4
num_workers = 2
nepochs = 20 #"Use it to change iterations"
weight_decay = 1e-1
best_loss = 1e+20 # number gotten from initial resnet18 run
stop_count = 4
print(f'batch_size = {batch_size}, learning_rate = {learning_rate} num_workers = {num_workers} , nepochs = {nepochs} , best_loss = {best_loss}, weight_decay={weight_decay}')
##-----------------------------------------------------------------------------------------------------------##

##-----------------------------------------------------------------------------------------------------------##
#------------ Data Loadinga and Pre-processing -------------------#
# Convert the data to a PyTorch tensor

##-----------------------------------------------------------------------------------------------------------##

##-------------------------------------------------GARBAGE CLASSIFICATION----------------------------------------------------------##
### Testing the Model
net = GarbageModel(4, (3,224,224), True)
net.to(device)
#------- Training Parameters ---------#
# Loss Function
criterion = nn.CrossEntropyLoss(weight = class_weights)  # Loss function
optimizer = torch.optim.AdamW(net.parameters(), lr = learning_rate, weight_decay=weight_decay) # Optimizer used for training
# Decay LR by a factor of 0.1 every 7 epochs
scheduler = ExponentialLR(optimizer, gamma=0.9)

##---------------------------Main------------##
PATH = './garbage_net.pth' # Path to save the best model
# e = []
# trainL= []
# valL =[]
counter = 0
print('Traning and Validation \n')
for epoch in range(nepochs):  # loop over the dataset multiple times
    # Training Loop
    net.train()
    train_loss = 0.0
    for i, batch in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        images = batch['image'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(images, input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    print(f'{epoch + 1},  train loss: {train_loss / i:.3f},', end = ' ')

    scheduler.step()

#---------------Validation----------------------------#
    net.eval()
    val_loss = 0.0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, batch in enumerate(valloader, 0):
            # get the inputs; data is a list of [inputs, labels] 
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            images = batch['image'].to(device)

            outputs = net(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
        print(f'val loss: {val_loss / i:.3f}')
        
        
    # valL.append(val_loss)
    # trainL.append(train_loss)
    # e.append(epoch)

    # Save best model
    if val_loss < best_loss:
        print("Saving model")
        torch.save(net.state_dict(), PATH)
        best_loss = val_loss
        counter = 0
        # Early stopping
    elif val_loss > best_loss:
        counter += 1
        if counter >= stop_count:
            print("Early stopping")
            break
    else:
        counter = 0

##------------------------------TESTING-----------------------------------------------------------------------------##
# Using the metrics function to evaluate the model.
print('Testing \n')
net = GarbageModel(4, (3,224,224), False)
net.load_state_dict(torch.load(PATH))
metrics_eval(net, testloader, device)
##-----------------------------------------------------------------------------------------------------------##