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

import os
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from network_trainer import NetworkTrainer

def do_save_func(epoch):
    return True

class SimpleMLPTrainer(NetworkTrainer):
    """Trainer for a simple fully connected network."""

    def __init__(self, train_loader, val_loader=None, criterion=None, optimizer=None, epochs=10, device='cpu', mi_method='binning', do_save_func=do_save_func, **kwargs):
        model = SimpleMLP()
        super(SimpleMLPTrainer, self).__init__(model, train_loader, val_loader, criterion, optimizer, epochs, device, mi_method, do_save_func=do_save_func, **kwargs)


# Define the MLP architecture
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 20)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(20, 20)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(20, 20)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(20, 10)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)
        return self.softmax(x)

if __name__ == '__main__':
    # Load dataset
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=10000, shuffle=False)

    # Train the model
    trainer = SimpleMLPTrainer(train_loader, val_loader, epochs=50, device='cuda', mi_method='binning2', verbose=True, do_save_func=do_save_func)
    trainer.train()

    # Save mutual information values
    # mi_values = trainer.get_mi_values()
    # if not os.path.exists('./results'):
    #     os.makedirs('./results')
    # np.savez('./results/simple_mlp_mi_values.npz', **mi_values)
    
    # Obtain the plotter and generate plots
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    if not os.path.exists('./plots/simple_mlp'):
        os.makedirs('./plots/simple_mlp')

    plotter = trainer.get_plotter()
    plotter.plot_information_plane(save_path=os.path.join('./plots/simple_mlp', 'information_plane.png'))
    plotter.plot_loss_accuracy(save_path=os.path.join('./plots/simple_mlp', 'loss_accuracy.png'))
    # for layer in range(4):
    #     plotter.plot_mutual_information_layer(layer_index=layer, save_path=os.path.join('./plots/simple_mlp', f'mutual_information_layer_{layer}.png'))