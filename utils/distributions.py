import tensorflow as tf
import numpy as np
from utils import support_functions
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import scipy.stats


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

        self.mu_n = self.mu.numpy()[0]
        self.sigma_n = self.sigma.numpy()

        self.dim = mu.get_shape()[1].value
        self.i_sigma = tf.constant(np.linalg.inv(np.copy(sigma)))

    def get_energy_fn(self):

        def energy_fn(x):
            return support_functions.gaussian_energy(x, self.mu, self.i_sigma)

        return energy_fn

    def get_density(self, x):
        exponent = -support_functions.gaussian_energy(x, self.mu, self.i_sigma)
        return tf.math.exp(exponent)

    def get_samples(self, n):
        return np.random.multivariate_normal(self.mu_n, self.sigma_n, n)

    def viz(self, n, show, save_path):
        assert self.dim == 2, 'Dimension must be 2.'
        samples = self.get_samples(n)
        plt.figure(figsize=(8,8))
        plt.scatter(samples[:, 0], samples[:, 1], s=5)
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path, format='png')
