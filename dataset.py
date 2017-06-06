import tensorflow as tf
import numpy as np
import logging
import os, glob, cv2

def _np_one_hot(x, n):
    y = np.zeros([len(x), n])
    y[np.arange(len(x)), x] = 1
    return y

def _read_and_decode(directory, s_t_shape, num_act, x_t_1_shape):
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
    a_t = tf.one_hot(a_t, num_act)

    return s_t, a_t, x_t_1

class Dataset(object):
    def __init__(self, directory, num_act, mean_path, num_threads=1, capacity=1e5, batch_size=32, scale=(1.0/255.0), s_t_shape=[84, 84, 12], x_t_1_shape=[84, 84, 3]):
        # Load image mean
        mean = np.load(os.path.join(mean_path))
        self.mean = mean
        
        # Prepare data flow
        s_t, a_t, x_t_1 = _read_and_decode(directory, 
                                        s_t_shape=s_t_shape,
                                        num_act=num_act,
                                        x_t_1_shape=x_t_1_shape)
        self.s_t_batch, self.a_t_batch, self.x_t_1_batch = tf.train.shuffle_batch([s_t, a_t, x_t_1],
                                                            batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=int(capacity*0.25),
                                                            num_threads=num_threads)
        # Subtract image mean (according to J Oh design)
        self.mean_const = tf.constant(mean, dtype=tf.float32)
        self.s_t_batch = (self.s_t_batch - tf.tile(self.mean_const, [1, 1, 4])) * scale
        self.x_t_1_batch = (self.x_t_1_batch - self.mean_const) * scale
        
    def __call__(self):        
        return {'s_t': self.s_t_batch,
                'a_t': self.a_t_batch,
                'x_t_1': self.x_t_1_batch,
                'mean': self.mean_const}

class CaffeDataset(object):
    '''
        Used to load data with directory structure in original paper
    '''
    def __init__(self, dir, num_act, mean_path, mode='tf', num_frame=4, num_channel=3):
        # dir: image data directory, each image should be named as %05d.png
        # num_act: number of action in action space (only support discrete action)
        # mean_path: mean image file path (NOTE: you must convert mean.binaryproto to npy file)
        # mode: tf or caffe (differ in s, a format)
        # num_frame: initial frame 
        # num_channel: number of channel per frame
        self.mean = np.load(mean_path)
        self.num_act = num_act
        self.dir = dir
        self.mode = mode
        self.num_frame = num_frame
        self.num_channel = num_channel
 
    def _process_frame_tf(self, s, img):
        # s: state np array
        # img: frame input
        s[:, :, :-self.num_channel] = s[:, :, self.num_channel:]
        s[:, :, -self.num_channel:] = img
        return s

    def _process_frame_caffe(self, s, img):
        raise NotImplemented()             
   
    def __call__(self, max_iter=5):
        with open(os.path.join(self.dir, 'act.log')) as act_log:
            cnt_frame = 0

            if self.mode == 'tf':
                s = np.zeros([84, 84, self.num_frame * self.num_channel])
            else:
                s = np.zeros([1, self.num_frame, self.num_channel, 84, 84], dtype=np.flaot32)

            a = np.zeros([self.num_frame, 1], dtype=np.int32)

            for filename in sorted(glob.glob(os.path.join(self.dir, '*.png')))[:max_iter]:
                img = cv2.imread(filename)

                if self.mode == 'tf':
                    s = self._process_frame_tf(s, img)
                else:
                    s = self._process_frame_caffe(s, img)

                a[:, :-1] = a[:, 1:]
                a[:, -1] = int(act_log.readline()[:-1])

                if cnt_frame < self.num_frame:
                    cnt_frame += 1
                else:
                    yield s, _np_one_hot(a[-1], self.num_act)

            

            
           

