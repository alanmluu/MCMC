class Profiler(object):

    def init(self):
        self.history = {}
        self.run_num = 0

    def start_run(self, name=None):
        self.run_num += 1
        if name is None:
            self.history['run{}'.format(self.run_num)] = []
        else:
            self.history[name] = []
