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
import torch
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
        print(p_xs, unique_inverse_x)
        
        bins = np.linspace(-1, 1, num_bins, dtype='float32')
        digitized_layer = bins[np.digitize(np.squeeze(layer_data.reshape(1, -1)), bins) - 1].reshape(len(layer_data), -1)
        print(digitized_layer)
        
        p_ts, _ = get_probabilities(digitized_layer)
        print(p_ts)
        
        h_layer = -np.sum(p_ts * np.log(p_ts + np.finfo(float).eps))  # Adding epsilon to avoid log(0)
        print(h_layer)
        h_layer_given_input = 0.
        for xval in np.arange(len(p_xs)):
            p_t_given_x, _ = get_probabilities(digitized_layer[unique_inverse_x == xval, :])
            h_layer_given_input += - p_xs[xval] * np.sum(p_t_given_x * np.log(p_t_given_x + np.finfo(float).eps))
            print(h_layer_given_input)

        return h_layer - h_layer_given_input

    def kde_mutual_info(self, X, Y, bandwidth):
        """
        Calculate mutual information using kernel density estimation.

        Parameters
        ----------
        X : array-like
            Input data.
        Y : array-like
            Output data.
        bandwidth : float
            Bandwidth for KDE.
        
        Returns
        -------
        mi : float
            Mutual information between X and Y.
        """
        raise NotImplementedError("Not implemented yet")
        # def compute_pairwise_distances(matrix):
        #     squared_sum = np.sum(np.square(matrix), axis=1, keepdims=True)
        #     dists = squared_sum + squared_sum.T - 2 * np.dot(matrix, matrix.T)
        #     return dists

        # def get_shape_info(tensor):
        #     dims = float(tensor.shape[1])
        #     N = float(tensor.shape[0])
        #     return dims, N

        # def estimate_entropy_kl(tensor, var):
        #     dims, N = get_shape_info(tensor)
        #     dists = compute_pairwise_distances(tensor)
        #     dists2 = dists / (2 * var)
        #     normconst = (dims / 2.0) * np.log(2 * np.pi * var)
        #     lprobs = logsumexp(-dists2, axis=1) - np.log(N) - normconst
        #     h = -np.mean(lprobs)
        #     return dims / 2 + h

        # def estimate_entropy_bd(tensor, var):
        #     dims, N = get_shape_info(tensor)
        #     val = estimate_entropy_kl(tensor, 4 * var)
        #     return val + np.log(0.25) * dims / 2

        # if isinstance(X, torch.Tensor):
        #     X = X.cpu().detach().numpy()
        # if isinstance(Y, torch.Tensor):
        #     Y = Y.cpu().detach().numpy()

        # var = bandwidth
        # hX = estimate_entropy_bd(X, var)
        # hY = estimate_entropy_bd(Y, var)
        # hXY = estimate_entropy_bd(np.hstack((X, Y)), var)

        # return hX + hY - hXY

    def mutual_info_kraskov(self, X, Y):
        """
        Calculate mutual information using Kraskov method.

        Parameters
        ----------
        X : array-like
            Input data.
        Y : array-like
            Output data.
        
        Returns
        -------
        mi : float
            Mutual information between X and Y.
        """
        raise NotImplementedError("Not implemented yet")
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)
        data = np.hstack((X, Y))

        n = len(X)
        d = 2

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

        Parameters
        ----------
        X : array-like or torch.Tensor
            Input data.
        Y : array-like or torch.Tensor
            Output data.
        
        Returns
        -------
        mi : float
            Mutual information between X and Y.
        """
        if isinstance(X, torch.Tensor):
            X = X.cpu().detach().numpy()
        if isinstance(Y, torch.Tensor):
            Y = Y.cpu().detach().numpy()

        if self.method == 'binning':
            return self.mutual_info_bin(X, Y, self.bins)
            # try:
            #     return self.mutual_info_bin(X, Y, self.bins)
            # except Exception as e:
            #     print(f"Error in binning method: {e}")
            #     return 0.0
        elif self.method == 'kde':
            try:
                return self.kde_mutual_info(X, Y, self.bandwidth)
            except Exception as e:
                print(f"Error in KDE method: {e}")
                return 0.0
        elif self.method == 'kraskov':
            try:
                return self.mutual_info_kraskov(X, Y)
            except Exception as e:
                print(f"Error in Kraskov method: {e}")
                return 0.0
        else:
            raise ValueError("Unknown method: choose from 'binning', 'kde', 'kraskov'")

# Example usage with neural network outputs
if __name__ == '__main__':
    X = np.random.rand(100,2)
    Y = X * 0.5 + np.random.rand(100,2) * 0.1
    
    mi_calculator = MutualInformationCalculator(method='binning')
    print(f"Mutual Information (Binning): {mi_calculator.calculate(X, Y)}")
    
    mi_calculator = MutualInformationCalculator(method='kde')
    print(f"Mutual Information (KDE): {mi_calculator.calculate(X, Y)}")
    
    mi_calculator = MutualInformationCalculator(method='kraskov')
    print(f"Mutual Information (Kraskov): {mi_calculator.calculate(X, Y)}")