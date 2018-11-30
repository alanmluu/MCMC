class Simulation(object):

    def __init__(self):
        self.systems = {}
        self.system_count = 0

    def add_system(self, system, name=None):
        self.system_count += 1
        if name is not None:
            self.systems[name] = system
        else:
            self.systems['system' + str(self.system_count)] = system

    def profiler
