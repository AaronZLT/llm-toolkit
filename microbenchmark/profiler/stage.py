from profiler.kernel import BenchKernel
from profiler.module import BenchModule

class BenchStage(object):
    def __init__(self, name, start_kernel_list,
                 end_kernel_list, stream, config):
        self.name = name
        self.start_kernel_list = start_kernel_list
        self.end_kernel_list = end_kernel_list
        self.stream = stream
        self.config = config
        self.start_time = []
        self.end_time = []
        self.during_time = []
        self.fine_modules = []
        self.coarse_modules = []
    
    def match(self):
        def check_match(idx, kernel_list):
            if len(self.stream.kernel_list) - idx < len(kernel_list):
                return False, 0
            for incre_ii in range(len(kernel_list)):
                if kernel_list[incre_ii] not in self.stream.kernel_list[ii+incre_ii]['name']:
                    return False, 0
            return True, len(kernel_list)-1

        skip = 0
        for ii in range(len(self.stream.kernel_list)):
            if skip != 0:
                skip -= 1
                continue
            flag, skip = check_match(ii, self.start_kernel_list)
            if flag == True:
                self.start_time.append(self.stream.kernel_list[ii]['ts'])
                continue
            flag, skip = check_match(ii, self.end_kernel_list)
            if flag == True:
                self.end_time.append(self.stream.kernel_list[ii+skip]['ts']+self.stream.kernel_list[ii+skip]['dur'])
        
        if self.name == "optimize":
            # Optimize stage
            del self.end_time[0]
            del self.start_time[-1]
        for ii in range(len(self.start_time)):
            self.during_time.append(self.end_time[ii] - self.start_time[ii])
        self.print_info(f"The {self.name} stage during time", self.during_time)

    def extract_fine_module(self, module_names):
        fine_module_kernel_dict = dict()
        for module_name in module_names:
            fine_module_kernel_dict[module_name] = self.config["fine_module"][module_name]

        def check_match_find_module(idx, name, kernel_name_list):
            kernel_list = []
            if len(self.stream.kernel_list) - idx < len(kernel_name_list):
                return False, 0, None
            for kernel_ii in range(len(kernel_name_list)):
                if kernel_name_list[kernel_ii] not in self.stream.kernel_list[idx+kernel_ii]['name']:
                    return False, 0, None
                kernel_list.append(BenchKernel(kernel_name_list[kernel_ii], self.stream.kernel_list[idx+kernel_ii]['ts'], self.stream.kernel_list[idx+kernel_ii]['dur']))
            return True, len(kernel_name_list)-1, BenchModule(name, kernel_list, self.stream)

        skip = 0
        for idx in range(len(self.stream.kernel_list)):
            if skip != 0:
                skip -= 1
                continue
            for module_name in module_names:
                flag, skip, module = check_match_find_module(idx, module_name, fine_module_kernel_dict[module_name])
                if flag == True:
                    self.fine_modules.append(module)
                    break
        
        sum_fine_modules_dur_time = dict()
        num_fine_modules_dur_time = dict()
        for module_name in module_names:
            sum_fine_modules_dur_time[module_name] = 0
            num_fine_modules_dur_time[module_name] = 0
        for fine_module in self.fine_modules:
            sum_fine_modules_dur_time[fine_module.name] += fine_module.during_time
            num_fine_modules_dur_time[fine_module.name] += 1
        
        self.print_info(f"{self.name} fine_module time", [f"{name}: {sum_fine_modules_dur_time[name]}, num: {num_fine_modules_dur_time[name]}, avg: {sum_fine_modules_dur_time[name]/num_fine_modules_dur_time[name]}" for name in module_names])

    def extract_coarse_module(self):
        coarse_module_names = self.config["stage_coarse_module"].get(self.name, None)
        if coarse_module_names == None:
            return
        
        fine_module_names = self.config["stage_fine_module"][self.name]
        self.extract_fine_module(fine_module_names)

        coarse_module_contains_module_dict = {}
        for coarse_module_name in coarse_module_names:
            coarse_module_contains_module_dict[coarse_module_name] = self.config["coarse_module"][coarse_module_name]
        

    def extract(self):
        self.match()
        # TODO
        # satistics the module time
        self.extract_coarse_module()
    
    def print_info(self, message, sub_message):
        print("[INFO] ", message)
        for message_ii in sub_message:
            print("\t", message_ii)
