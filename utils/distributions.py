import tensorflow as tf
import numpy as np
import support_functions
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class Distribution(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_density(self, x):
        pass


class Gaussian(object):

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

        self.dim = len(mu)
        self.i_sigma = np.linalg.inv(np.copy(sigma))

    def get_energy_fn(self):

        def energy_fn(x):
            return support_functions.gaussian_energy(x, self.mu, self.i_sigma)

        return energy_fn

    def get_samples(self, n):
        return np.random.multivariate_normal(self.mu, self.sigma, n)

    def visualize(self, n, show=False, save_path=None):
        assert self.dim == 2, 'Dimension must be 2.'
        samples = self.get_samples(n)
        plt.scatter(samples[:, 0], samples[:, 1])
        if show:
            plt.show()
        elif save_path is not None:
            plt.savefig(save_path, format='png')
