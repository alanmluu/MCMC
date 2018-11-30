from distributions import Distribution

class System(object):

    def __init__(self,
                 ensemble=None,
                 walker=None,
                 kernel=None,
                 distribution=None,
                 initializer=None):

        self.ensemble = ensemble
        self.walker = walker
        self.kernel = kernel
        self.distribution = distribution
        self.initializer = initializer

    @property
    def dim(self):
        return self.__dim

    @dim.setter
    def dim(self, dim):
        if isinstance(self.distribution, Distribution):
            return self.distribution.dim

    def initialize_walker(self):
        self.initializer.initialize(self.walker)

    def step(self):

        curr_x = self.walker.get_position()
        prop_x = self.kernel.propose(curr_x, self.distribution)

        curr_x_dens = self.distribution.get_density(curr_x)
        prop_x_dens = self.distribution.get_density(prop_x)

        trans_factor = self.proposal.get_trans_factor(curr_x, prop_x)
        ensemble_factor = self.ensemble.get_ensemble_factor(curr_x_dens,
                                                            prop_x_dens)

        accept_ratio = min(1, ensemble_factor * trans_factor)

        self.walker.walk(accept_ratio, prop_x)

    def evolve(self, n):
        for i in range(n):
            self.step()
