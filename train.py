import tensorflow as tf
import numpy as np

import argparse
import sys
import logging

from model import ActionConditionalVideoPredictionModel
from dataset import Dataset

def get_config(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config

def main(args):
    # Create dataset
    logging.info('Create data flow from %s' % args.train)
    train_data = Dataset(args.train)

    # Create model
    logging.info('Create model for training [lr = %f, epochs = %d, batch_size = %d]' % (args.lr, args.epoch, args.batch_size) )
    model = ActionConditionalVideoPredictionModel(train_data(), 
                                                optimizer_args={'lr': args.lr})
    
    # Create initializer
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    # Create data producer threads
    coord = tf.train.Coordinator()

    # Get optimizer operation and loss opearation from model
    train_op = model.train
    loss_op = model.loss
   
    # Config session
    config = get_config(args)
    
    # Start session
    with tf.Session(config=config) as sess:
        sess.run(init)
        threads = train_data.start(sess=sess, coord=coord)
        for epoch in range(args.epoch):
            sess.run([train_op])
            loss = sess.run([loss_op])[0]
            logging.info('Epoch %d: L2 loss = %f' % (epoch, loss))
            pass
        coord.request_stop()
        coord.join(threads)
    
    
if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', help='training data directory', type=str, default='example')
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--epoch', help='epoch', type=int, default=3)
    parser.add_argument('--batch_size', help='batch size', type=int, default=4)
    args = parser.parse_args()

    main(args)



