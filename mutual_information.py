################################################################################
# Title:            mutual_information.py                                      #
# Description:      Calculate mutual information using different methods.      #
# Author:           Aidin Attar                                                #
# Date:             2024-07-01                                                 #
# Version:          0.1                                                        #
# Usage:            python mutual_information.py                               #
# Notes:            None                                                       #
# Python version:   3.11.7                                                     #
################################################################################

import numpy as np
import scipy.stats as stats
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mutual_info_score
import torch
from scipy.special import digamma

import numpy as np
import scipy.stats as stats
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import mutual_info_score
import torch
from scipy.special import digamma

class MutualInformationCalculator:
    """Class to calculate mutual information using different methods."""

    def __init__(self, method='binning', bins=30, bandwidth=0.1, k=3):
        """
        Initialize the MutualInformationCalculator.

        Parameters:
        method (str): Method to calculate mutual information ('binning', 'kde', 'kraskov').
        bins (int): Number of bins for the binning method (default: 30).
        bandwidth (float): Bandwidth for the KDE method (default: 0.1).
        k (int): Number of nearest neighbors for the Kraskov method (default: 3).
        """
        self.method = method
        self.bins = bins
        self.bandwidth = bandwidth
        self.k = k

    def mutual_info_bin(self, X, Y, nbins=30, bin_max=1, bin_min=-1):
        """
        Calculate mutual information by binning the data.

        Parameters:
        X (array-like): Input data.
        Y (array-like): Output data.
        
        Returns:
        float: Mutual information between X and Y.
        """
        binsize = (bin_max - bin_min) / nbins

        # Flatten the arrays if they are not already 1-dimensional
        if X.ndim > 1:
            X = X.ravel()
        if Y.ndim > 1:
            Y = Y.ravel()

        # Digitize the input and output data
        X_digitized = np.floor(X / binsize).astype('int')
        Y_digitized = np.floor(Y / binsize).astype('int')

        # Ensure that the lengths match
        min_length = min(len(X_digitized), len(Y_digitized))
        X_digitized = X_digitized[:min_length]
        Y_digitized = Y_digitized[:min_length]

        # Calculate the mutual information
        c_xy = np.histogram2d(X_digitized, Y_digitized, bins=nbins)[0]
        mi = mutual_info_score(None, None, contingency=c_xy)
        return mi

    def mutual_info_kde(self, X, Y):
        """
        Calculate mutual information using kernel density estimation.

        Parameters:
        X (array-like): Input data.
        Y (array-like): Output data.
        
        Returns:
        float: Mutual information between X and Y.
        """
        # Flatten the arrays if they are not already 1-dimensional
        if X.ndim > 1:
            X = X.ravel()
        if Y.ndim > 1:
            Y = Y.ravel()
        
        # Estimate densities
        kde_X = stats.gaussian_kde(X, bw_method=self.bandwidth)
        kde_Y = stats.gaussian_kde(Y, bw_method=self.bandwidth)
        kde_XY = stats.gaussian_kde(np.vstack([X, Y]), bw_method=self.bandwidth)
        
        # Evaluate densities
        p_X = kde_X(X)
        p_Y = kde_Y(Y)
        p_XY = kde_XY(np.vstack([X, Y]))
        
        # Ensure no zero probabilities
        p_X = np.maximum(p_X, 1e-10)
        p_Y = np.maximum(p_Y, 1e-10)
        p_XY = np.maximum(p_XY, 1e-10)
        
        # Calculate mutual information
        mi = np.mean(np.log(p_XY / (p_X * p_Y)))
        return mi

    def mutual_info_kraskov(self, X, Y):
        """
        Calculate mutual information using the Kraskov method.

        Parameters:
        X (array-like): Input data.
        Y (array-like): Output data.
        k (int): Number of nearest neighbors.
        
        Returns:
        float: Mutual information between X and Y.
        """
        # Ensure inputs are 2-dimensional
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        data = np.hstack((X, Y))
        
        n = len(X)
        d = data.shape[1]  # Dimensionality
        
        nbrs = NearestNeighbors(n_neighbors=self.k+1, metric='chebyshev').fit(data)
        distances, _ = nbrs.kneighbors(data)
        
        epsilon = distances[:, self.k].reshape(n, 1)
        
        nbrs_x = NearestNeighbors(n_neighbors=self.k, metric='chebyshev').fit(X)
        nbrs_y = NearestNeighbors(n_neighbors=self.k, metric='chebyshev').fit(Y)
        
        nx = np.array([len(nbrs_x.radius_neighbors([x], radius=epsilon[i], return_distance=False)[0]) - 1 for i, x in enumerate(X)])
        ny = np.array([len(nbrs_y.radius_neighbors([y], radius=epsilon[i], return_distance=False)[0]) - 1 for i, y in enumerate(Y)])
        
        mi = digamma(self.k) - np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1)) + digamma(n)
        return mi

    def calculate(self, X, Y):
        """
        Calculate mutual information using the specified method.

        Parameters:
        X (array-like or torch.Tensor): Input data.
        Y (array-like or torch.Tensor): Output data.
        
        Returns:
        float: Mutual information between X and Y.
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.cpu().detach().numpy()
        
        if self.method == 'binning':
            return self.mutual_info_bin(X, Y)
        elif self.method == 'kde':
            return self.mutual_info_kde(X, Y)
        elif self.method == 'kraskov':
            return self.mutual_info_kraskov(X, Y)
        else:
            raise ValueError("Unknown method: choose from 'binning', 'kde', 'kraskov'")


# Example usage with neural network outputs
if __name__ == '__main__':
    X = np.random.rand(1000)
    Y = X * 0.5 + np.random.rand(1000) * 0.1
    
    mi_calculator = MutualInformationCalculator(method='binning')
    print(f"Mutual Information (Binning): {mi_calculator.calculate(X, Y)}")
    
    mi_calculator = MutualInformationCalculator(method='kde')
    print(f"Mutual Information (KDE): {mi_calculator.calculate(X, Y)}")
    
    mi_calculator = MutualInformationCalculator(method='kraskov')
    print(f"Mutual Information (Kraskov): {mi_calculator.calculate(X, Y)}")