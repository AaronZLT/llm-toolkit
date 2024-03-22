
class BenchModule(object):
    def __init__(self, name, kernel_list, stream, start_idx=0, end_idx=-1) -> None:
        self.name = name
        self.kernel_list = kernel_list
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.stream = stream
        self.start_time = self.kernel_list[start_idx].start_time
        self.end_time = self.kernel_list[end_idx].end_time
        self.during_time = self.end_time - self.start_time
    