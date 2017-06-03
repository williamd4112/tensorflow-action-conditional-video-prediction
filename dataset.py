import tensorflow as tf
import numpy as np
import logging
import os

def _read_and_decode(directory, s_t_shape, x_t_1_shape):
    filenames = tf.train.match_filenames_once('./%s/*.tfrecords' % (directory))
    filename_queue = tf.train.string_input_producer(filenames)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                       'a_t': tf.FixedLenFeature([], tf.int64),
                                       's_t' : tf.FixedLenFeature([], tf.string),
                                       'x_t_1' : tf.FixedLenFeature([], tf.string),
                                       })

    s_t = tf.decode_raw(features['s_t'], tf.uint8)
    x_t_1 = tf.decode_raw(features['x_t_1'], tf.uint8)
    
    s_t = tf.reshape(s_t, s_t_shape)
    x_t_1 = tf.reshape(x_t_1, x_t_1_shape)

    s_t = tf.cast(s_t, tf.float32)
    x_t_1 = tf.cast(x_t_1, tf.float32)
    a_t = tf.cast(features['a_t'], tf.int32)
    a_t = tf.expand_dims(a_t, 0)

    return s_t, a_t, x_t_1

class Dataset(object):
    def __init__(self, directory, mean_path, batch_size=32, s_t_shape=[84, 84, 12], x_t_1_shape=[84, 84, 3]):
        # Load image mean
        mean = np.load(os.path.join(mean_path))
        
        # Prepare data flow
        s_t, a_t, x_t_1 = _read_and_decode(directory, 
                                        s_t_shape=s_t_shape, 
                                        x_t_1_shape=x_t_1_shape)
        self.s_t_batch, self.a_t_batch, self.x_t_1_batch = tf.train.shuffle_batch([s_t, a_t, x_t_1],
                                                            batch_size=batch_size, capacity=2000,
                                                            min_after_dequeue=100)
        # Subtract image mean (according to J Oh design)
        self.s_t_batch = self.s_t_batch - np.tile(mean, [4]) 
        self.x_t_1_batch = self.x_t_1_batch - mean

    def __call__(self):        
        return {'s_t': self.s_t_batch,
                'a_t': self.a_t_batch,
                'x_t_1': self.x_t_1_batch}
            
           
