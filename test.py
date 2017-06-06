import tensorflow as tf
import numpy as np
import cv2

import argparse
import sys, os
import logging

import cPickle as pickle

from model import ActionConditionalVideoPredictionModel
from dataset import Dataset

def get_config(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config

import re
def load_caffe_model(path):
    tf_ops = []
    with tf.variable_scope('', reuse=True) as scope:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            for key in data:
                val = data[key]
                var = tf.get_variable(key)
                tf_ops.append(tf.assign(var, data[key]))
                logging.info('%s loaded with shape %s' % (key, val.shape))
    return tf.group(*tf_ops)
             

def main(args):
    with tf.Graph().as_default() as graph:
        # Create dataset
        logging.info('Create data flow from %s' % args.data)
        data = Dataset(directory=args.data, 
                        num_act=args.num_act, 
                        mean_path=args.mean, 
                        batch_size=args.batch_size, num_threads=1, capacity=100)
    
        # Create model
        logging.info('Create model from %s' % (args.load))
        model = ActionConditionalVideoPredictionModel(inputs=data(), num_act=args.num_act, is_train=False)

        # Create initializer
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        # Create weight load operation
        load_op = load_caffe_model(args.load)
         
        # Config session
        config = get_config(args)
        
        # Setup summary
        test_summary_op = tf.summary.merge_all('test')
        test_summary_writer = tf.summary.FileWriter(os.path.join(args.log, 'test'), graph)

        def post_process(data, mean, scale):
            mean = np.transpose(mean, [2, 0, 1])
            data = np.transpose(data, [2, 0, 1])
            t = data.copy().squeeze()
            t /= scale
            t += mean
            t = t.clip(0, 255)
            return t.astype('uint8').squeeze().transpose([1, 0, 2]).transpose([0, 2, 1])

        # Start session
        with tf.Session(config=config) as sess:
            coord = tf.train.Coordinator()
            logging.info('Initializing')
            sess.run(init)
            logging.info('Loading')
            sess.run(load_op)
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in range(10):
                pred_data, summary = sess.run([model.output, test_summary_op])
                test_summary_writer.add_summary(summary, i)
                cv2.imwrite('%03d.png' % i, post_process(pred_data[0], data.mean, 1.0/255.0))
            coord.request_stop()
            coord.join(threads)        
    
if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', help='summary directory', type=str, default='caffe-test')
    parser.add_argument('--data', help='testing data directory', type=str, default='example/test')
    parser.add_argument('--mean', help='image mean path', type=str, required=True)
    parser.add_argument('--load', help='caffe-dumped model path', type=str, required=True)
    parser.add_argument('--num_act', help='num acts', type=int, required=True)
    parser.add_argument('--batch_size', help='batch size', type=int, default=1)
    args = parser.parse_args()

    main(args)



