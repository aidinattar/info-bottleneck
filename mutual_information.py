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

import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma, logsumexp

class MutualInformationCalculator:
    """Class to calculate mutual information using different methods."""

    def __init__(self, method='binning', bins=30, bandwidth=0.1, k=3):
        self.method = method
        self.bins = bins
        self.bandwidth = bandwidth
        self.k = k

    def mutual_info_bin(self, input_data, layer_data, num_bins):
        """
        Calculate mutual information using binning.

        Parameters
        ----------
        input_data : array-like
            Input data.
        layer_data : array-like
            Output data.
        num_bins : int
            Number of bins for binning method.
        
        Returns
        -------
        float : Mutual information between input_data and layer_data.
        """
        def get_probabilities(data):
            unique_ids = np.ascontiguousarray(data).view(np.dtype((np.void, data.dtype.itemsize * data.shape[1])))
            _, unique_inverse, unique_counts = np.unique(unique_ids, return_index=False, return_inverse=True, return_counts=True)
            return np.asarray(unique_counts / float(sum(unique_counts))), unique_inverse

        if input_data.ndim == 1:
            input_data = input_data.reshape(-1, 1)
        if layer_data.ndim == 1:
            layer_data = layer_data.reshape(-1, 1)

        p_xs, unique_inverse_x = get_probabilities(input_data)
        bins = np.linspace(-1, 1, num_bins, dtype='float32') 
        digitized = bins[np.digitize(np.squeeze(layer_data.reshape(1, -1)), bins) - 1].reshape(len(layer_data), -1)
        p_ts, _ = get_probabilities(digitized)
        
        h_layer = -np.sum(p_ts * np.log(p_ts + np.finfo(float).eps))  # Adding epsilon to avoid log(0)
        h_layer_given_input = 0.
        for xval in np.arange(len(p_xs)):
            p_t_given_x, _ = get_probabilities(digitized[unique_inverse_x == xval, :])
            h_layer_given_input += - p_xs[xval] * np.sum(p_t_given_x * np.log(p_t_given_x + np.finfo(float).eps))
        
        return h_layer - h_layer_given_input

    def mutual_info_kde(self, X, Y):
        def compute_pairwise_distances(matrix):
            squared_sum = np.sum(np.square(matrix), axis=1, keepdims=True)
            dists = squared_sum + squared_sum.T - 2 * np.dot(matrix, matrix.T)
            return dists

        def get_shape_info(tensor):
            dims = float(tensor.shape[1])
            N = float(tensor.shape[0])
            return dims, N

        def estimate_entropy_kl(tensor, var):
            dims, N = get_shape_info(tensor)
            dists = compute_pairwise_distances(tensor)
            dists2 = dists / (2 * var)
            normconst = (dims / 2.0) * np.log(2 * np.pi * var)
            lprobs = logsumexp(-dists2, axis=1) - np.log(N) - normconst
            h = -np.mean(lprobs)
            return dims / 2 + h

        def estimate_entropy_bd(tensor, var):
            dims, N = get_shape_info(tensor)
            val = estimate_entropy_kl(tensor, 4 * var)
            return val + np.log(0.25) * dims / 2

        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.cpu().detach().numpy()

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        var = self.bandwidth
        hX = estimate_entropy_bd(X, var)
        hY = estimate_entropy_bd(Y, var)
        hXY = estimate_entropy_bd(np.hstack((X, Y)), var)

        return hX + hY - hXY

    def mutual_info_kraskov(self, X, Y):
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
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.cpu().detach().numpy()
        
        if X.shape[1] != Y.shape[1]:
            raise ValueError("Input data and layer data must have the same number of features.")
        
        mi_values = []
        for i in range(X.shape[1]):
            if self.method == 'binning':
                mi_values.append(self.mutual_info_bin(X[:, i], Y[:, i], self.bins))
            elif self.method == 'kde':
                mi_values.append(self.mutual_info_kde(X[:, i], Y[:, i]))
            elif self.method == 'kraskov':
                mi_values.append(self.mutual_info_kraskov(X[:, i], Y[:, i]))
            else:
                raise ValueError("Unknown method: choose from 'binning', 'kde', 'kraskov'")
        
        return np.mean(mi_values)