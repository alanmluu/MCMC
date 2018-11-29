from abc import ABC, abstractmethod


class System(object):

    def __init__(self, ensemble, walker, kernel, distribution, initializer):
        self.ensemble = ensemble
        self.walker = walker
        self.kernel = kernel
        self.distribution = distribution
        self.initializer = initializer
        self.dim = self.distribution.dim

    def initialize(self):
        self.initializer.initialize(self.walker)

    def step(self):

        curr_x = self.walker.get_position()
        prop_x = self.kernel.propose(self.walker)

        curr_x_dens = self.distribution.get_density(curr_x)
        prop_x_dens = self.distribution.get_density(prop_x)

        trans_factor = self.proposal.get_trans_factor(curr_x, prop_x)
        ens_factor = self.ensemble.get_ens_factor(curr_x_dens, prop_x_dens)

        hastings_factor = min(1, ens_factor * trans_factor)

        self.walker.walk(hastings_factor, prop_x)

    def evolve(self, n):
        for i in range(n):
            self.step()
