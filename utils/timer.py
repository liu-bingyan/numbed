import time
import numpy as np
from argparse import Namespace


class Timer:

    def __init__(self):
        self.save_times = []
        self.start_time = 0
 
    def start(self):
        self.start_time = time.process_time()

    def end(self):
        end_time = time.process_time()
        self.save_times.append(end_time - self.start_time)
         
    def get_average_time(self):
        return np.mean(self.save_times)
    def get_total_time(self):
        return np.sum(self.save_times)
    
    def get_results(self):
        return {}

class Test_Timer(Timer):
    def __init__(self,args=None):
        super().__init__()
    def get_results(self):
        result_dict = {
            'testing_time' : self.get_average_time()
        }
        return result_dict
    
class Train_Timer(Timer):
    def __init__(self,args):
        super().__init__()
        self.save_epochs = []
        self.save_times = []
        self.args = args

    def end(self, batches_count = 0):
        end_time = time.process_time()
        self.save_times.append(end_time - self.start_time)
        self.save_epochs.append(batches_count * self.args.batch_size / self.args.num_samples if batches_count!=0 else 1)
       
    def get_average_epochs(self):
        return int(np.mean(self.save_epochs))
    def get_total_epochs(self):
        return np.sum(self.save_epochs)
    def get_average_time_per_epoch(self):
        return self.get_total_time()/self.get_total_epochs()
    def get_results(self):
        result_dict = {
            'trainning_time' : self.get_average_time(),
            'average_epochs' : self.get_average_epochs(),
            'time_per_epoch': self.get_average_time_per_epoch()
        }
        return result_dict
    
if __name__ == "__main__":
    args = Namespace(num_samples=1000,batch_size=100)
    timer = Train_Timer(args)
    for i in range(10):
        timer.start()
        for j in range(int(10e6)):
            pass
        timer.end(100)
    print(timer.get_results())