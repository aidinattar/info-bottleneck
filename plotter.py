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

import os
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
    """Class to handle plotting of mutual information and training statistics."""

    def __init__(self, mi_values, train_losses, val_losses, train_accuracies, val_accuracies):
        self.mi_values = mi_values
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.train_accuracies = train_accuracies
        self.val_accuracies = val_accuracies

    def plot_information_plane(self, save_path=None):
        """
        Plot the information plane I(X;T) vs I(T;Y) with points connected within the same epoch.

        Parameters
        ----------
        save_path : str, optional
            Path to save the plot.
        """
        plt.figure(figsize=(10, 6))
        
        epochs = self.mi_values['epochs']
        I_XT = np.array(self.mi_values['I(X;T)'])
        I_TY = np.array(self.mi_values['I(T;Y)'])

        # Create a scatter plot with color mapping to epoch
        scatter = plt.scatter(I_XT.flatten(), I_TY.flatten(), c=np.repeat(epochs, I_XT.shape[1]), cmap='viridis', s=10)
        plt.colorbar(scatter, label='Epoch')
        
        # Plot points and lines connecting points within the same epoch
        colormap = plt.cm.viridis
        for epoch in range(len(epochs)):
            plt.plot(I_XT[epoch], I_TY[epoch], marker='o', linestyle='-', color=colormap(epoch / len(epochs)))

        plt.xlabel('I(X;T)')
        plt.ylabel('I(T;Y)')
        plt.title('Information Plane')
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.show()

    def plot_loss_accuracy(self, save_path=None):
        """
        Plot the training and validation loss and accuracy.

        Parameters
        ----------
        save_path : str
            Path to save the plot.
        """
        epochs = range(1, len(self.train_losses) + 1)
        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color=color)
        ax1.plot(epochs, self.train_losses, 'r-', label='Train Loss')
        if self.val_losses:
            ax1.plot(epochs, self.val_losses, 'r--', label='Val Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Accuracy', color=color)
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Train Accuracy')
        if self.val_accuracies:
            ax2.plot(epochs, self.val_accuracies, 'b--', label='Val Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.legend(loc='upper right')

        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.show()

    def plot_mutual_information_layer(self, layer_index=0, save_path=None):
        """
        Plot mutual information for a specific layer across epochs.

        Parameters
        ----------
        layer_index : int
            Index of the layer to plot.
        save_path : str
            Path to save the plot.
        """
        epochs = range(1, len(self.mi_values['epochs']) + 1)
        I_XT_layer = [mi[layer_index] for mi in self.mi_values['I(X;T)']]
        I_TY_layer = [mi[layer_index] for mi in self.mi_values['I(T;Y)']]

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, I_XT_layer, label=f'I(X;T) - Layer {layer_index + 1}')
        plt.plot(epochs, I_TY_layer, label=f'I(T;Y) - Layer {layer_index + 1}')
        plt.xlabel('Epoch')
        plt.ylabel('Mutual Information')
        plt.title(f'Mutual Information for Layer {layer_index + 1}')
        plt.legend()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.show()