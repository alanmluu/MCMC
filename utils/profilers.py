import numpy as np


class Profiler(object):

    def __init__(self):
        self.history = {}
        self.run_num = 0
        self.cur_run_name = None
        self.cur_run = None

    def start_run(self, x, run_name=None):
        self.run_num += 1
        if run_name:
            self.cur_run_name = run_name
        else:
            self.cur_run_name = 'run{}'.format(self.run_num)
        self.cur_run = x
        self.history[self.cur_run_name] = self.cur_run

    def log(self, x):
        self.cur_run = np.vstack((self.cur_run, x))
        self.history[self.cur_run_name] = self.cur_run
