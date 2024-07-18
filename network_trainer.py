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

import json
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from mutual_information import MutualInformationCalculator
from plotter import Plotter


class NetworkTrainer:
    """Parent class for training neural networks and calculating mutual information."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        criterion=None,
        optimizer=None,
        epochs=10,
        device='cuda',
        mi_method='binning',
        activation_path='activations.json',
        mi_values_path='mi_values.json',
        verbose=True,
        save_dir='results',
        do_save_func=None,
        lr=0.001,
        momentum=0.9,
        save_activation=False,
        optimizer_name='sgd'
    ):
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
            Device to use for training (default: 'cuda').
        mi_method : str, optional
            Method to calculate mutual information ('binning', 'kde', 'kraskov').
        save_dir : str, optional
            Directory to save training logs.
        do_save_func : callable, optional
            Function that returns True if data should be saved on the current epoch.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion or nn.CrossEntropyLoss()
        if optimizer_name == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        self.optimizer = optimizer or optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        self.epochs = epochs
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.mi_calculator = MutualInformationCalculator(method=mi_method)
        self.mi_values = {'I(X;T)': [], 'I(T;Y)': [], 'epochs': []}
        self.activations = {}
        self.activation_path = os.path.join(save_dir, activation_path)
        self.mi_values_path = os.path.join(save_dir, mi_values_path)
        self.verbose = verbose
        self.save_dir = save_dir
        self.do_save_func = do_save_func
        self.save_activation = save_activation
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Initialize weights
        self.model.apply(self.weights_init)

    def weights_init(self, m):
        """Initialize the weights of the model."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def register_hooks(self):
        """Register hooks to capture activations of each layer."""

        def get_activation(name):
            def hook(model, input, output):
                self.activations[name] = output.detach().cpu().numpy()
            return hook

        hooks = []
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.ReLU, nn.Tanh, nn.Softmax)):  # Register hooks for Linear and ReLU layers
                hooks.append(layer.register_forward_hook(get_activation(name)))
        return hooks

    def save_activations(self):
        """Save activations to file."""
        with open(self.activation_path, 'w') as f:
            json.dump({k: v.tolist() for k, v in self.activations.items()}, f)

    def save_mi_values(self):
        """Save mutual information values to file."""
        with open(self.mi_values_path, 'w') as f:
            json.dump(self.mi_values, f)

    def save_epoch_data(self, epoch, loss, grad_data):
        """Save epoch data to file."""
        fname = os.path.join(self.save_dir, f"epoch{epoch:08d}.json")
        print(f"\tSaving {fname}")
        data = {
            'epoch': epoch,
            'loss': loss,
            'grad_data': grad_data,
            'activations': {k: v.tolist() for k, v in self.activations.items()}
        }
        with open(fname, 'w') as f:
            json.dump(data, f)

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
            # take the time before the epoch starts
            start_time = time.time()
            self.train_epoch()
            if self.val_loader:
                self.validate_epoch()
            # take the time after the epoch ends
            end_time = time.time()
            print(f'Epoch {epoch+1}/{self.epochs},\n'
                  f'\tTime: {end_time - start_time:.2f}s, '
                  f'\tTrain Loss: {self.train_losses[-1]:.4f}, '
                  f'\tTrain Accuracy: {self.train_accuracies[-1]:.2f}%, '
                  f'\tVal Loss: {self.val_losses[-1] if self.val_losses else "N/A":.4f}, '
                  f'\tVal Accuracy: {self.val_accuracies[-1] if self.val_accuracies else "N/A":.2f}%')
            self.calculate_mutual_information(epoch)
            if self.save_activation:
                self.save_activations()
            self.save_mi_values()
            if self.save_activation:
                if self.do_save_func and self.do_save_func(epoch):
                    grad_data = self.collect_gradients()
                    loss = {
                        'train_loss': self.train_losses[-1],
                        'val_loss': self.val_losses[-1] if self.val_losses else None,
                        'train_accuracy': self.train_accuracies[-1],
                        'val_accuracy': self.val_accuracies[-1] if self.val_accuracies else None,
                    }
                    self.save_epoch_data(epoch, loss, grad_data)

            end_time = time.time()
            print(f"\tTime taken for epoch {epoch+1}: {end_time - start_time:.2f}s")

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
            for data, target in self.val_loader:
                data = data.to(self.device)
                self.model(data)  # Forward pass to trigger hooks
                break  # Only need a single batch to get the activations

        # Save indexes of test data for each of the output classes
        saved_labelixs = {}
        for i in range(10):
            saved_labelixs[i] = target.cpu().numpy() == i

        # Remove hooks
        for hook in hooks:
            hook.remove()

        I_XT = []
        I_TY = []

        input_data = data.cpu().numpy().reshape(data.size(0), -1)  # Flatten the input data
        target_data = target.cpu().numpy().reshape(-1, 1)  # Reshape target data to match batch size

        if self.verbose:
            print(f"Debug: Input data shape: {input_data.shape}, Target data shape: {target_data.shape}")  # Debug statement

        for layer_name, activation in self.activations.items():
            activations_flattened = activation.reshape(activation.shape[0], -1)  # Flatten activations
            if self.verbose:
                print(f"Debug: Layer: {layer_name}, Activation shape: {activations_flattened.shape}")  # Debug statement
            
            # mi_input = self.mi_calculator.calculate(input_data, activations_flattened)
            # mi_output = self.mi_calculator.calculate(activations_flattened, target_data)
            mi_input, mi_output = self.mi_calculator.calculate(saved_labelixs, activations_flattened)
            I_XT.append(mi_input)
            I_TY.append(mi_output)
            if self.verbose:
                print(f"Layer: {layer_name}, MI Input: {mi_input}, MI Output: {mi_output}")  # Debug statement
        
        self.mi_values['I(X;T)'].append(I_XT)
        self.mi_values['I(T;Y)'].append(I_TY)
        self.mi_values['epochs'].append(epoch)

        if self.verbose:
            print(f"Epoch {epoch}, I(X;T): {I_XT}, I(T;Y): {I_TY}")  # Debug statement

    def collect_gradients(self):
        """Collect gradients for the current model."""
        self.model.zero_grad()
        grad_data = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_data[name] = param.grad.data.cpu().numpy().tolist()
        return grad_data

    def get_plotter(self):
        """Return a Plotter instance initialized with the training data."""
        return Plotter(self.mi_values, self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies)

    def get_mi_values(self):
        """Return the mutual information values."""
        return self.mi_values