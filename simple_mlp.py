################################################################################
# Title:            simple_mlp.py                                              #
# Description:      A simple fully connected network for MNIST classification. #
# Author:           Aidin Attar                                                #
# Date:             2024-07-01                                                 #
# Version:          0.1                                                        #
# Usage:            python simple_mlp.py                                       #
# Notes:            None                                                       #
# Python version:   3.11.7                                                     #
################################################################################

import torch.nn as nn
from network_trainer import NetworkTrainer

class SimpleMLPTrainer(NetworkTrainer):
    """Trainer for a simple fully connected network."""

    def __init__(self, train_loader, val_loader=None, activation='relu', criterion=None, optimizer=None, epochs=10, device='cpu', mi_method='binning', do_save_func=None, **kwargs):
        model = SimpleMLP(activation=activation)
        super(SimpleMLPTrainer, self).__init__(model, train_loader, val_loader, criterion, optimizer, epochs, device, mi_method, do_save_func=do_save_func, **kwargs)


# Define the MLP architecture
# class SimpleMLP(nn.Module):
#     def __init__(self, activation='relu'):
#         super(SimpleMLP, self).__init__()
#         self.fc1 = nn.Linear(28*28, 1024)
#         self.act1 = nn.ReLU() if activation == 'relu' else nn.Tanh()
#         self.fc2 = nn.Linear(1024, 200)
#         self.act2 = nn.ReLU() if activation == 'relu' else nn.Tanh()
#         self.fc3 = nn.Linear(200, 150)
#         self.act3 = nn.ReLU() if activation == 'relu' else nn.Tanh()
#         self.fc4 = nn.Linear(150, 100)
#         self.act4 = nn.ReLU() if activation == 'relu' else nn.Tanh()
#         self.fc5 = nn.Linear(100, 50)
#         self.act5 = nn.ReLU() if activation == 'relu' else nn.Tanh()
#         self.fc6 = nn.Linear(50, 10)
#         self.softmax = nn.Softmax(dim=1)
    
#     def forward(self, x):
#         x = x.view(-1, 28*28)  # Flatten the input
#         x = self.act1(self.fc1(x))
#         x = self.act2(self.fc2(x))
#         x = self.act3(self.fc3(x))
#         x = self.act4(self.fc4(x))
#         x = self.act5(self.fc5(x))
#         x = self.fc6(x)
#         return self.softmax(x)

# class SimpleMLP(nn.Module):
#     def __init__(self, activation='relu'):
#         super(SimpleMLP, self).__init__()
#         self.fc1 = nn.Linear(28*28, 512)
#         self.act1 = nn.ReLU() if activation == 'relu' else nn.Tanh()
#         self.fc2 = nn.Linear(512, 256)
#         self.act2 = nn.ReLU() if activation == 'relu' else nn.Tanh()
#         self.fc3 = nn.Linear(256, 128)
#         self.act3 = nn.ReLU() if activation == 'relu' else nn.Tanh()
#         self.fc4 = nn.Linear(128, 64)
#         self.act4 = nn.ReLU() if activation == 'relu' else nn.Tanh()
#         self.fc5 = nn.Linear(64, 32)
#         self.act5 = nn.ReLU() if activation == 'relu' else nn.Tanh()
#         self.fc6 = nn.Linear(32, 16)
#         self.act6 = nn.ReLU() if activation == 'relu' else nn.Tanh()
#         self.fc7 = nn.Linear(16, 10)
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, x):
#         x = x.view(-1, 28*28)
#         x = self.act1(self.fc1(x))
#         x = self.act2(self.fc2(x))
#         x = self.act3(self.fc3(x))
#         x = self.act4(self.fc4(x))
#         x = self.act5(self.fc5(x))
#         x = self.act6(self.fc6(x))
#         x = self.fc7(x)
#         return self.softmax(x)


# Define the MLP architecture
class SimpleMLP(nn.Module):
    def __init__(self, activation='relu'):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 1024)
        self.act1 = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.fc2 = nn.Linear(1024, 20)
        self.act2 = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.fc3 = nn.Linear(20, 20)
        self.act3 = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.fc4 = nn.Linear(20, 20)
        self.act4 = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.fc5 = nn.Linear(20, 20)
        self.act5 = nn.ReLU() if activation == 'relu' else nn.Tanh()
        self.fc6 = nn.Linear(20, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.act3(self.fc3(x))
        x = self.act4(self.fc4(x))
        x = self.act5(self.fc5(x))
        x = self.fc6(x)
        return self.softmax(x)