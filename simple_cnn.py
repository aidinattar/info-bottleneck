################################################################################
# Title:            simple_cnn.py                                              #
# Description:      A simple convolutional network for MNIST classification.   #
# Author:           Aidin Attar                                                #
# Date:             2024-07-01                                                 #
# Version:          0.1                                                        #
# Usage:            python simple_cnn.py                                       #
# Notes:            None                                                       #
# Python version:   3.11.7                                                     #
################################################################################

import torch.nn as nn
from network_trainer import NetworkTrainer

class SimpleCNNTrainer(NetworkTrainer):
    """Trainer for a simple convolutional network."""

    def __init__(self, train_loader, val_loader=None, activation='relu', criterion=None, optimizer=None, epochs=10, device='cpu', mi_method='binning', do_save_func=None, **kwargs):
        model = SimpleCNN(activation=activation)
        super(SimpleCNNTrainer, self).__init__(model, train_loader, val_loader, criterion, optimizer, epochs, device, mi_method, do_save_func=do_save_func, **kwargs)


# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, activation='relu'):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64*7*7, 128)
        self.act3 = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = x.view(-1, 64*7*7)  # Flatten the tensor
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)