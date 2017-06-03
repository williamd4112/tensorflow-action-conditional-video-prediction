import tensorflow as tf
import numpy as np
import cv2

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
    train_data = Dataset(directory=args.train, mean_path=args.mean, batch_size=args.batch_size)
    test_data = Dataset(directory=args.test, mean_path=args.mean, batch_size=args.batch_size)
    
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
        train_summary_writer = tf.summary.FileWriter(args.log, graph=sess.graph)
        test_summary_writer = tf.summary.FileWriter(args.log, graph=sess.graph)
        summary_op = tf.summary.merge_all()

        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for epoch in range(args.epoch):
            # Train
            sess.run([train_op])
            train_loss, train_summary = sess.run([loss_op, summary_op])
            train_summary_writer.add_summary(train_summary, epoch)

            if (epoch + 1) % args.show_per_epoch == 0:
                logging.info('Epoch %d: Training L2 loss = %f' % (epoch, train_loss))

            # Test
            if (epoch + 1) % args.test_per_epoch == 0:
                test_loss = sess.run([loss_op])[0]             
                logging.info('Epoch %d: Testing L2 loss = %f' % (epoch, test_loss))
                test_loss, test_summary = sess.run([loss_op, summary_op])
                test_summary_writer.add_summary(test_summary, epoch)
           
        coord.request_stop()
        coord.join(threads)
    
    
if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', help='summary directory', type=str, default='log')
    parser.add_argument('--train', help='training data directory', type=str, default='example/train')
    parser.add_argument('--test', help='testing data directory', type=str, default='example/test')
    parser.add_argument('--mean', help='image mean path', type=str, default='example/mean.npy')
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--epoch', help='epoch', type=int, default=10000)
    parser.add_argument('--show_per_epoch', help='epoch', type=int, default=20)
    parser.add_argument('--test_per_epoch', help='epoch', type=int, default=20)
    parser.add_argument('--batch_size', help='batch size', type=int, default=32)
    args = parser.parse_args()

    main(args)



