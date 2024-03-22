
class BenchKernel(object):
    def __init__(self, name, ts, dur):
        self.name = name
        self.start_time = ts
        self.during_time = dur
        self.end_time = ts + dur