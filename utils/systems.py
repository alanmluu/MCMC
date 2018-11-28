from abc import ABC, abstractmethod


class Ensemble(ABC):

    def __init__(self, walker, proposal, distribution, initializer):
        self.walker = walker
        self.proposal = proposal
        self.distribution = distribution
        self.initializer = initializer
        self.dim = self.distribution.dim

    def initialize(self):
        self.initializer.initialize(self.walker)


class CanonicalEnsemble(object):


    def hastings_accept_prob(prop_x, curr_x):


    def step(self):
        curr_x = self.walker.get_position()
        prop_x = self.proposal.propose(self.walker)

        curr_x_density = self.distribution.get_density(curr_x_density)
        prop_x_density = self.distribution.get_density(prop_x_density)

        trans_prob = proposal.

    def evolve(self, n):
        for i in range(n):
            proposal = self.proposer.propose(self.walker)
            self.acceptor()
