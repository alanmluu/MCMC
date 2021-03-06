from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf


class Kernel(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def propose(self):
        pass

class GaussianKernel(Kernel):

    def __init__(self, sigma):
        self.sigma = sigma
        self.sigma_n = sigma.numpy()

    def propose(self, x, distribution):
        x_n = x.numpy()[0]
        proposal_n = np.random.multivariate_normal(mean=x_n,
                                                    cov=self.sigma_n,
                                                    size=1)[0]
        return tf.constant([proposal_n], dtype="float32")

    def get_trans_factor(self, curr_x, prop_x):
        return tf.constant([1], dtype="float32")


class Hamiltonian(Kernel):

    def __init__(self, steps, eps, sigma, T=1):
        self.steps = steps
        self.eps = eps
        self.sigma = sigma
        self.sigma_n = sigma.numpy()
        #self.T = tf.constant([T], dtype='float32')
        self.T = 1

    def set_temp(self, T):
        self.T = tf.constant([T], dtype='float32')

    def get_gradient(self, x, distribution, residual):
        if residual is None:
            energy_fn = distribution.get_energy_fn()
            with tf.GradientTape() as g:
                g.watch(x)
                y = tf.scalar_mul(1/self.T, energy_fn(x))
            gradient = g.gradient(y, x)
            return gradient
        elif residual == 0:
            dist_den = distribution.get_density_fn()
            with tf.GradientTape() as g:
                g.watch(x)
                y = -tf.log(dist_den(x) + 1)
            gradient = g.gradient(y, x)
            return gradient
        else:
            dist_den = distribution.get_density_fn()
            res_den = residual.get_density_fn()
            with tf.GradientTape() as g:
                g.watch(x)
                y = -tf.log(dist_den(x) - res_den(x) + 1)
            gradient = g.gradient(y, x)
            return gradient

    def leapfrog_step(self, x, p, distribution, residual):
        p_int = p - ((self.eps/2) * self.get_gradient(x, distribution, residual))
        x_f = x + self.eps * p_int
        p_f = p_int - ((self.eps/2) * self.get_gradient(x_f, distribution, residual))
        return x_f, p_f

    def hamiltonian_flow(self, x, distribution, residual):
        p = np.random.multivariate_normal(mean=np.zeros(distribution.dim),
                                          cov=self.sigma_n,
                                          size=1)[0]
        for i in range(self.steps):
            x, p = self.leapfrog_step(x, p, distribution, residual)
        return x


    def propose(self, x, distribution, residual):
        return self.hamiltonian_flow(x, distribution, residual)

    def get_trans_factor(self, curr_x, prop_x):
        return tf.constant([1], dtype="float32")
