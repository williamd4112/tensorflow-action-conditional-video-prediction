import tensorflow as tf
import numpy as np
import cv2

import argparse
import sys, os
import logging

import cPickle as pickle

from model import ActionConditionalVideoPredictionModel
from dataset import Dataset, CaffeDataset

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
        caffe_dataset = CaffeDataset(dir=args.data, num_act=args.num_act, mean_path=args.mean)
        
        # Create model
        logging.info('Create model from %s' % (args.load))
        model = ActionConditionalVideoPredictionModel(inputs=None, num_act=args.num_act, is_train=False)

        # Create initializer
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        # Create weight load operation
        load_op = load_caffe_model(args.load)
         
        # Config session
        config = get_config(args)
        
        # Setup summary
        '''
        tf.summary.image('mean', tf.expand_dims(data.mean_const, 0), collections=['test'])
        test_summary_op = tf.summary.merge_all('test')
        test_summary_writer = tf.summary.FileWriter(os.path.join(args.log, 'test'), graph)
        '''
        # Start session
        with tf.Session(config=config) as sess:
            logging.info('Initializing')
            sess.run(init)
            logging.info('Loading')
            sess.run(load_op)

            for s, a in caffe_dataset():
                pred_data = sess.run([model.output], feed_dict={model.inputs['s_t']: [s],
                                                                model.inputs['a_t']: a})[0]
                print pred_data.shape
    
if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', help='summary directory', type=str, default='caffe-test')
    parser.add_argument('--data', help='testing data directory', type=str, required=True)
    parser.add_argument('--mean', help='image mean path', type=str, required=True)
    parser.add_argument('--load', help='caffe-dumped model path', type=str, required=True)
    parser.add_argument('--num_act', help='num acts', type=int, required=True)
    parser.add_argument('--batch_size', help='batch size', type=int, default=1)
    args = parser.parse_args()

    main(args)



