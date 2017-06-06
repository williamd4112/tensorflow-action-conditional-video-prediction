import tensorflow as tf
import numpy as np
import cv2

import argparse
import sys, os
import logging

from model import ActionConditionalVideoPredictionModel
from dataset import Dataset

def get_config(args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config

def main(args):
    with tf.Graph().as_default() as graph:
        # Create dataset
        logging.info('Create data flow from %s' % args.train)
        train_data = Dataset(directory=args.train, num_act=args.num_act, mean_path=args.mean, batch_size=args.batch_size, num_threads=4, capacity=10000)
        #test_data = Dataset(directory=args.test, num_act=args.num_act, mean_path=args.mean, batch_size=args.test_batch_size, num_threads=1, capacity=100)
    
        # Create model
        logging.info('Create model for training [lr = %f, epochs = %d, batch_size = %d]' % (args.lr, args.epoch, args.batch_size) )
        model = ActionConditionalVideoPredictionModel(inputs=train_data(), num_act=args.num_act, optimizer_args={'lr': args.lr})
        #test_model = ActionConditionalVideoPredictionModel(test_data(), is_train=False)
        # Create initializer
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        
        # Get optimizer operation and loss opearation from model
        train_op = model.train
        loss_op = model.loss
        global_step_var = model.global_step
        
        # Config session
        config = get_config(args)
        
        # Setup summary
        train_summary_op = tf.summary.merge_all('train')
        test_summary_op = tf.summary.merge_all('test')
        test_summary_writer = tf.summary.FileWriter(os.path.join(args.log, 'test'), graph)

        # Setup supervisor
        sv = tf.train.Supervisor(logdir=os.path.join(args.log, 'train'),
                init_op=init,
                graph=graph,
                summary_op=train_summary_op,
                global_step=global_step_var)
        
        # Start session
        with sv.managed_session(config=config) as sess:
            sv.start_queue_runners(sess)
            for epoch in range(args.epoch): 
                if (epoch) % args.show_per_epoch == 0:
                    _, train_loss, train_summary, global_step = sess.run([train_op, loss_op, train_summary_op, global_step_var])
                    logging.info('Epoch %d: Training L2 loss = %f' % (global_step, train_loss))
                    sv.summary_computed(sess, train_summary)
                else:
                    sess.run([train_op])
                '''      
                if (epoch) % args.test_per_epoch == 0:
                    test_loss, test_summary, global_step = sess.run([test_model.loss, test_summary_op, global_step_var])
                    logging.info('Epoch %d: Testing L2 loss = %f' % (global_step, test_loss))
                    test_summary_writer.add_summary(test_summary, (global_step))
                '''
            sv.request_stop()
        
    
if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', help='summary directory', type=str, default='example/log')
    parser.add_argument('--train', help='training data directory', type=str, default='example/train')
    parser.add_argument('--test', help='testing data directory', type=str, default='example/test')
    parser.add_argument('--mean', help='image mean path', type=str, default='example/mean.npy')
    parser.add_argument('--num_act', help='num acts', type=int, required=True)
    parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
    parser.add_argument('--epoch', help='epoch', type=int, default=15000000)
    parser.add_argument('--show_per_epoch', help='epoch', type=int, default=1000)
    parser.add_argument('--test_per_epoch', help='epoch', type=int, default=2000)
    parser.add_argument('--batch_size', help='batch size', type=int, default=32)
    parser.add_argument('--test_batch_size', help='batch size', type=int, default=64)
    args = parser.parse_args()

    main(args)



