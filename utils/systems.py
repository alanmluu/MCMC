from utils.distributions import Distribution
import matplotlib.pyplot as plt
from utils.support_functions import COLOR_CYCLE
import tensorflow as tf


class System(object):

    def __init__(self,
                 ensemble=None,
                 walker=None,
                 kernel=None,
                 distribution=None,
                 initializer=None,
                 profiler=None):

        self.ensemble = ensemble
        self.walker = walker
        self.kernel = kernel
        self.distribution = distribution
        self.initializer = initializer
        self.profiler = profiler

    @property
    def dim(self):
        return self.__dim

    @dim.setter
    def dim(self, dim):
        if isinstance(self.distribution, Distribution):
            return self.distribution.dim

    def initialize_walker(self, name=None):
        self.initializer.initialize(self.walker)
        self.profiler.start_run(self.walker.x, name)


    def step(self):
        assert self.walker.x is not None, "Walker position must be initialized."

        curr_x = self.walker.x
        prop_x = self.kernel.propose(curr_x, self.distribution)

        curr_x_dens = self.distribution.get_density(curr_x)
        prop_x_dens = self.distribution.get_density(prop_x)

        trans_factor = self.kernel.get_trans_factor(curr_x, prop_x)
        ensemble_factor = self.ensemble.get_ensemble_factor(curr_x_dens,
                                                            prop_x_dens)

        accept_ratio = tf.minimum(1, tf.multiply(trans_factor,
                                                 ensemble_factor)).numpy()[0]

        self.walker.walk(accept_ratio, prop_x)
        self.profiler.log(self.walker.x)

    def evolve(self, n):
        for i in range(n):
            self.step()

    def viz_dist(self, n=1000, show=True, save_path=None):
        self.distribution.viz(n, show, save_path)

    def viz_trajectory(self, run_name=None, n=1000, show=True, save_path=None):
        self.viz_dist(n, show=False, save_path=None)
        if run_name:
            X = self.profiler.history[run_name].numpy()
        else:
            X = self.profiler.cur_run.numpy()
        plt.plot(X[:,0], X[:,1], color=COLOR_CYCLE[1], marker="o")
        if show:
            plt.show()
        if save_path:
            plt.savefig(save_path, format='png')
