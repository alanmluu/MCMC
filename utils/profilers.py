import numpy as np
import tensorflow as tf


class Profiler(object):

    def __init__(self):
        self.history = {}
        self.accept_history = {}
        self.run_num = 0
        self.cur_run_name = None
        self.cur_run = None
        self.cur_accept = None

    def start_run(self, x, run_name=None):
        self.run_num += 1
        if run_name:
            self.cur_run_name = run_name
        else:
            self.cur_run_name = 'run{}'.format(self.run_num)
        self.cur_run = x
        self.history[self.cur_run_name] = self.cur_run
        self.cur_accept = []
        self.accept_history[self.cur_run_name] = self.cur_accept

    def log(self, x, accept):
        self.cur_run = tf.concat([self.cur_run, x], axis=0)
        self.history[self.cur_run_name] = self.cur_run
        self.cur_accept.append(accept)
        self.accept_history[self.cur_run_name] = self.cur_accept

    def get_runs(self):
        return self.history.keys()

    def remove_run(self, run):
        self.history.pop(run)

    def reset(self):
        self.__init__()

    def clean(self):
        for key in list(self.history.keys()):
            if key != self.cur_run_name:
                self.history.pop(key)
                self.accept_history.pop(key)
