################################################################################
# Title:            network_trainer.py                                         #
# Description:      Parent class for training neural networks and calculating  #
#                   mutual information.                                        #
# Author:           Aidin Attar                                                #
# Date:             2024-07-01                                                 #
# Version:          0.1                                                        #
# Usage:            python network_trainer.py                                  #
# Notes:            None                                                       #
# Python version:   3.11.7                                                     #
################################################################################

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from mutual_information import MutualInformationCalculator
from plotter import Plotter


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

class NetworkTrainer:
    """Parent class for training neural networks and calculating mutual information."""

    def __init__(self, model, train_loader, val_loader=None, criterion=None, optimizer=None, epochs=10, device='cpu', mi_method='binning'):
        """
        Initialize the NetworkTrainer.

        Parameters
        ----------
        model : nn.Module
            The neural network model to train.
        train_loader : DataLoader
            DataLoader for the training data.
        val_loader : DataLoader, optional
            DataLoader for the validation data.
        criterion : loss function, optional
            Loss function for training.
        optimizer : optimizer, optional
            Optimizer for training.
        epochs : int, optional
            Number of epochs to train the model.
        device : str, optional
            Device to use for training (default: 'cpu').
        mi_method : str, optional
            Method to calculate mutual information ('binning', 'kde', 'kraskov').
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer or optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        self.epochs = epochs
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.mi_calculator = MutualInformationCalculator(method=mi_method)
        self.mi_values = {'I(X;T)': [], 'I(T;Y)': [], 'epochs': []}
        self.activations = {}

    def _hook(self, layer_name):
        def hook(module, input, output):
            self.activations[layer_name] = output
        return hook

    def register_hooks(self):
        """Register hooks to capture activations for specific layers."""
        hooks = []
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                hooks.append(layer.register_forward_hook(self._hook(name)))
        return hooks

    def train_epoch(self):
        """Train the model for one epoch."""
        self.model.train()
        correct = 0
        total = 0
        running_loss = 0.0
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_accuracy = 100. * correct / total
        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_accuracy)

    def validate_epoch(self):
        """Validate the model for one epoch."""
        self.model.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                running_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_accuracy = 100. * correct / total
        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_accuracy)

    def train(self):
        """Train the model."""
        for epoch in range(self.epochs):
            self.train_epoch()
            if self.val_loader:
                self.validate_epoch()
            print(f'Epoch {epoch+1}/{self.epochs},\n'
                  f'\tTrain Loss: {self.train_losses[-1]:.4f}, '
                  f'\tTrain Accuracy: {self.train_accuracies[-1]:.2f}%, '
                  f'\tVal Loss: {self.val_losses[-1] if self.val_losses else "N/A":.4f}, '
                  f'\tVal Accuracy: {self.val_accuracies[-1] if self.val_accuracies else "N/A":.2f}%')
            self.calculate_mutual_information(epoch)

    def calculate_mutual_information(self, epoch):
        """
        Calculate mutual information for each layer and store the results.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """
        self.model.eval()
        self.activations = {}
        hooks = self.register_hooks()

        with torch.no_grad():
            for data, _ in self.train_loader:
                data = data.to(self.device)
                self.model(data)  # Forward pass to trigger hooks
                break  # Only need a single batch to get the activations

        # Remove hooks
        for hook in hooks:
            hook.remove()

        I_XT = []
        I_TY = []

        input_data = data.cpu().numpy().reshape(data.size(0), -1)  # Flatten the input data

        for layer_name, activation in self.activations.items():
            activations_flattened = activation.cpu().detach().numpy().reshape(activation.size(0), -1)  # Flatten activations
            mi_input = self.mi_calculator.calculate(input_data, activations_flattened)
            mi_output = self.mi_calculator.calculate(activations_flattened, input_data)
            I_XT.append(mi_input)
            I_TY.append(mi_output)
        
        self.mi_values['I(X;T)'].append(I_XT)
        self.mi_values['I(T;Y)'].append(I_TY)
        self.mi_values['epochs'].append(epoch)

    def get_plotter(self):
        """Return a Plotter instance initialized with the training data."""
        return Plotter(self.mi_values, self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies)

    def get_mi_values(self):
        """Return the mutual information values."""
        return self.mi_values