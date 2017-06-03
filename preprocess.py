import numpy as np
import tensorflow as tf
import os, sys, cv2

import glob
from tqdm import *

class EpisodeReader(object):
    def __init__(self, path, height=84, width=84):
        self.reader = tf.python_io.tf_record_iterator(path=path)
        self.height = height
        self.width = width
    
    def read(self):
        for string_record in self.reader:
            example = tf.train.Example()
            example.ParseFromString(string_record)
            
            a_t = int(example.features.feature['a_t']
                                         .int64_list
                                         .value[0]) 
           
            s_t_string = (example.features.feature['s_t']
                                          .bytes_list
                                          .value[0])

            x_t_1_string = (example.features.feature['x_t_1']
                                          .bytes_list
                                          .value[0])
                       
            s_t_raw = np.fromstring(s_t_string, dtype=np.uint8)
            s_t = s_t_raw.reshape((self.height, self.width, -1))

            x_t_1_raw = np.fromstring(x_t_1_string, dtype=np.uint8)
            x_t_1 = x_t_1_raw.reshape((self.height, self.width, -1))
            
            yield s_t, a_t, x_t_1

if __name__ == '__main__':
    path = sys.argv[1]
    
    mean = np.zeros([84, 84, 3], dtype=np.float32)
    n = 0
    for path in tqdm(glob.glob(os.path.join(path, '*.tfrecords'))):
        reader = EpisodeReader(path)
        for s, a, x in (reader.read()):
            for i in range(0, 12, 3):
                mean += s[:,:,i:i+3]
                n += 1
    mean /= n
    np.save(sys.argv[2], mean)
 
