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

class SimpleMLPTrainer(NetworkTrainer):
    """Trainer for a simple fully connected network."""

    def __init__(self, train_loader, val_loader=None, criterion=None, optimizer=None, epochs=10, device='cpu', mi_method='binning'):
        model = SimpleMLP()
        super(SimpleMLPTrainer, self).__init__(model, train_loader, val_loader, criterion, optimizer, epochs, device, mi_method)


# Define the MLP architecture
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return self.softmax(x)

if __name__ == '__main__':
    # Load dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Train the model
    trainer = SimpleMLPTrainer(train_loader, val_loader, epochs=5, device='cuda', mi_method='binning')
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
    for layer in range(4):
        plotter.plot_mutual_information_layer(layer_index=layer, save_path=os.path.join('./plots/simple_mlp', f'mutual_information_layer_{layer}.png'))