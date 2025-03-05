# Important Packages

# Shows the Accuracy, f1-score and confusion matrix
#Import packages
import torch # pytorch main library
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import glob
import torchvision # computer vision utilities
from torchvision import *

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import os
import re
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import seaborn as sns

##-----------------------------------------------------------------------------------------------------------##
#-------------- Metrics -------------#
# Recall
#Precision
# F1 Score
# Accuracy

# F1- SCORE
def metrics_eval(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids']#.to(device)
            attention_mask = batch['attention_mask']#.to(device)
            labels = batch['label']#.to(device)
            images = batch['image']#.to(device)

            outputs = model(images, input_ids, attention_mask)
           
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # append true and predicted labels
            y_true.append(labels.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())

                # calculate macro F1 score
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')

        # calculate micro F1 score 
        f1_micro = f1_score(y_true, y_pred, average='micro')
        print(f'F1-SCORE of the netwwork is given ass micro: {f1_micro}, macro: {f1_macro}')
        
        # ACCURACY
        accuracy = 100*accuracy_score(y_true, y_pred, normalize=True)
        print(f'Accuracy of the network on the test images: {accuracy} %')
        
        # #CONFUSION MATRIX
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()
##-----------------------------------------------------------------------------------------------------------##

# Explainability

# Generalizability

# 