import json
from profiler.stage import BenchStage
from profiler.stream import BenchStream

class BenchTracer(object):
    def __init__(self, config, raw_data) -> None:
        self.config = config
        self.raw_data = raw_data
        self.stream_list = []
        self.stage_list = []
    
    def extract_stream(self):
        stream7_kernel = []
        stream20_kernel = []
        stream84_kernel = []
        for meta_data in self.raw_data['traceEvents']:
            if meta_data.get('args') == None:
                continue
            stream = meta_data['args'].get('stream')
            if stream == None:
                continue
            # magic num 7, 20 and 84 extracted from trace file.
            if stream == 7:
                stream7_kernel.append(meta_data)
            elif stream == 20:
                stream20_kernel.append(meta_data)
            else: stream84_kernel.append(meta_data)
        
        def sort_key(module):
            return module['ts'] 

        stream7_kernel.sort(key=sort_key)
        stream20_kernel.sort(key=sort_key)
        stream84_kernel.sort(key=sort_key)
        stream7 = BenchStream("stream7", stream7_kernel)
        stream20 = BenchStream("stream20", stream20_kernel)
        stream84 = BenchStream("stream84", stream84_kernel)
        
        self.stream_list.append(stream7)
        self.stream_list.append(stream20)
        self.stream_list.append(stream84)

        self.print_info("stream kerenl num", [f"{stream.name} {len(stream.kernel_list)}" for stream in self.stream_list])
    
    def extract_stage(self):
        forward_start_kernel_list = self.config['stage']['forward']["start"]
        forward_end_kernel_list = self.config['stage']['forward']["end"]
        backward_start_kernel_list = self.config['stage']['backward']["start"]
        backward_end_kernel_list = self.config['stage']['backward']["end"]
        optimize_start_kernel_list = self.config['stage']['optimize']["start"]
        optimize_end_kernel_list = self.config['stage']['optimize']["end"]

        # The name of self.stream[0] is stream7, which contains of most computation kernels. 
        # TODO, this should be tuned by config
        forward_stage = BenchStage("forward", forward_start_kernel_list, forward_end_kernel_list, self.stream_list[0], self.config)
        backward_stage = BenchStage("backward", backward_start_kernel_list, backward_end_kernel_list, self.stream_list[0], self.config)
        optimize_stage = BenchStage("optimize", optimize_start_kernel_list, optimize_end_kernel_list, self.stream_list[0], self.config)

        forward_stage.extract()
        backward_stage.extract()
        optimize_stage.extract()

        self.stage_list.append(forward_stage)
        self.stage_list.append(backward_stage)
        self.stage_list.append(optimize_stage)


    def extract(self):
        self.extract_stream()
        self.extract_stage()
    
    def print_info(self, message, sub_message):
        print("[INFO] ", message)
        for message_ii in sub_message:
            print("\t" , message_ii)

