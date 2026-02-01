import sys
import time
import numpy as np
import json


class Auxiliary(object):
    def __init__(self):
        pass

    def fail(self, msg):
        named_tuple = time.localtime()  # get struct_time
        time_string = time.strftime("%Y.%m.%d %H:%M:%S", named_tuple)
        f = open('logged.txt', 'a')
        self.println("[" + time_string + "]" + " " + msg, f)  # print to file
        f.close()
        exit(1)

    def println(self, s, f=sys.stdout):
        if sys.version_info[0] < 3:
            print >> f, s
        else:
            func = eval('print')
            func(s, end='\n', file=f)

    def log(self, msg):
        named_tuple = time.localtime()  # get struct_time
        time_string = time.strftime("%Y.%m.%d %H:%M:%S", named_tuple)
        f = open('logged.txt', 'a')
        self.println("[" + time_string + "]" + " " + msg, f)  # print to file
        f.close()

    def convert_ndarray_to_list(self, obj):
        if isinstance(obj, dict):
            return {key: self.convert_ndarray_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_ndarray_to_list(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj