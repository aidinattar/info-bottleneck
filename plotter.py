################################################################################
# Title:           plotter.py                                                  #
# Description:     Class for plotting mutual information dynamics and training #
# Author:          Aidin Attar                                                 #
# Date:            2024-07-01                                                  #
# Version:         0.1                                                         #
# Usage:           python plotter.py                                           #
# Notes:           None                                                        #
# Python version:  3.11.7                                                      #
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12


class Plotter:
    """Class to handle plotting of mutual information values."""

    def __init__(self, mi_values, train_losses, val_losses, train_accuracies, val_accuracies):
        """
        Initialize the Plotter.

        Parameters
        ----------
        mi_values : dict
            Dictionary containing mutual information values.
        train_losses : list
            List of training losses over epochs.
        val_losses : list
            List of validation losses over epochs.
        train_accuracies : list
            List of training accuracies over epochs.
        val_accuracies : list
            List of validation accuracies over epochs.
        """
        self.mi_values = mi_values
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.train_accuracies = train_accuracies
        self.val_accuracies = val_accuracies

    def plot_information_plane(self, save_path=None):
        """Plot the information plane showing I(X;T) and I(T;Y) for each layer and epoch."""
        I_XT = np.array(self.mi_values['I(X;T)'])
        I_TY = np.array(self.mi_values['I(T;Y)'])
        epochs = np.array(self.mi_values['epochs'])
        num_layers = I_XT.shape[1]

        plt.figure(figsize=(10, 8))
        for layer in range(num_layers):
            plt.scatter(I_XT[:, layer], I_TY[:, layer], c=epochs, cmap='viridis', label=f'Layer {layer+1}', edgecolors='k', s=50)

        plt.colorbar(label='Epoch')
        plt.xlabel('I(X;T)')
        plt.ylabel('I(T;Y)')
        plt.title('Information Plane')
        plt.legend()
        # plt.show()
        if save_path:
            plt.savefig(save_path)

    def plot_loss_accuracy(self, save_path=None):
        """Plot the training and validation loss and accuracy over epochs."""
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(12, 6))

        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label='Training Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label='Training Accuracy')
        plt.plot(epochs, self.val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()

        plt.tight_layout()
        # plt.show()
        if save_path:
            plt.savefig(save_path)

    def plot_mutual_information_layer(self, layer_index, save_path=None):
        """Plot the mutual information I(X;T) and I(T;Y) for a specific layer over epochs.

        Parameters
        ----------
        layer_index : int
            Index of the layer to plot mutual information for.
        """
        I_XT = np.array(self.mi_values['I(X;T)'])[:, layer_index]
        I_TY = np.array(self.mi_values['I(T;Y)'])[:, layer_index]
        epochs = np.array(self.mi_values['epochs'])

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, I_XT, label='I(X;T)')
        plt.plot(epochs, I_TY, label='I(T;Y)')
        plt.xlabel('Epochs')
        plt.ylabel('Mutual Information')
        plt.title(f'Mutual Information for Layer {layer_index + 1}')
        plt.legend()
        if save_path:
            plt.savefig(save_path)