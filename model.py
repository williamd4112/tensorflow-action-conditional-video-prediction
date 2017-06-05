import tensorflow as tf
import numpy as np
import re

from tf_ops import ReLu, Conv2D, FC, Deconv2D

NUM_CHANNELS = 3
NUM_FRAMES = 4

class ActionConditionalVideoPredictionModel(object):
    def __init__(self, num_act, inputs=None, is_train=True, optimizer_args=None):
        # optimizer_args: optimizer arguments (e.g. optimizer type, learning rate, ...) (dict)
        self.is_train = is_train
        self.num_act = num_act
        self.optimizer_args = optimizer_args
        self._create_input(inputs)
        self._create_model()
        self._create_output()
        self._create_loss()
        if self.is_train:
            self._create_optimizer()

    def _create_input(self, inputs):
        if inputs == None:
            self.inputs = {'s_t': tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, (NUM_CHANNELS * NUM_FRAMES)]),
                       'a_t': tf.placeholder(dtype=tf.int32, shape=[None, self.num_act]),
                       'x_t_1': tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, (NUM_CHANNELS)]),
                        'mean': tf.placeholder(dtype=tf.float32, shape=[84, 84, NUM_CHANNELS])}
        else:
            self.inputs = inputs
 
    def _create_model(self):
        self.encode = self._create_encoder(self.inputs['s_t'])
        self.act_embed = self._create_action_embedding(self.inputs['a_t'])
        self.decode = self._create_decoder(self.encode, self.act_embed)

    def _create_output(self):
        self.output = self.decode
        output_img = tf.cast((self.output * 255.0) + self.inputs['mean'], tf.uint8)
        ground_truth_img = tf.cast((self.inputs['x_t_1'] * 255.0) + (self.inputs['mean']), tf.uint8)

        tf.summary.image('pred', output_img, collections=['train', 'test']) 
        tf.summary.image('ground', ground_truth_img, collections=['train', 'test']) 

    def _create_loss(self): 
        with tf.variable_scope('loss', reuse=not self.is_train) as scope:
            t = self.inputs['x_t_1']
            self.loss = tf.reduce_mean(tf.nn.l2_loss(self.output - t, name='l2'))
            tf.summary.scalar("loss", self.loss, collections=['train', 'test'])
        
        tf.summary.image('x_pred_t_1', tf.cast(self.decode * 255.0, tf.uint8), collections=['train']) 
        tf.summary.image('x_t_1', tf.cast(t * 255.0, tf.uint8), collections=['train']) 

    def _create_optimizer(self):
        with tf.variable_scope('optimize', reuse=not self.is_train) as scope:
            # Setup global_step, optimizer
            self.global_step = tf.get_variable('global_step', shape=(), initializer=tf.constant_initializer(0.0), trainable=False)

            lr = self.optimizer_args['lr']

            self.learning_rate = tf.train.exponential_decay(lr, self.global_step, 1e5, 0.9, staircase=True)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, name='optimizer')
            #self.train = self.optimizer.minimize(self.loss, global_step=self.global_step)
            
            # Compute gradient for weights, bias
            grads_vars = self.optimizer.compute_gradients(self.loss)
            bias_pattern = re.compile('.*/b')
            grads_vars_mult = []
            for grad, var in grads_vars:
                if bias_pattern.match(var.op.name):
                    grads_vars_mult.append((grad * 2.0, var))
                else: 
                    grads_vars_mult.append((grad, var))
                tf.summary.histogram('grad_%s' % (var.op.name), grad, collections=['train'])
            grads_clip = [(tf.clip_by_value(grad, -0.1, 0.1), var) for grad, var in grads_vars_mult]
            self.train = self.optimizer.apply_gradients(grads_clip, global_step=self.global_step)
            
            tf.summary.scalar("learning_rate", self.learning_rate, collections=['train'])

    def _create_encoder(self, x):
        # x: input image (tensor([batch_size, 84, 84, 12]))
        with tf.variable_scope('enc', reuse=not self.is_train) as scope:
            l = Conv2D(x, [6, 6], 64, 2, 'VALID', 'conv1')
            l = ReLu(l, 'relu1')
            l = Conv2D(l, [6, 6], 64, 2, 'SAME', 'conv2')
            l = ReLu(l, 'relu2')
            l = Conv2D(l, [6, 6], 64, 2, 'SAME', 'conv3')
            l = ReLu(l, 'relu3')
            l = FC(l, 1024, 'fc1')
            l = ReLu(l, 'relu4')
            l = FC(l, 2048, 'fc2', initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
        return l

    def _create_action_embedding(self, act):
        # act: action input (tensor([batch_size, num_act])) (one-hot vector)
        with tf.variable_scope('act-embed', reuse=not self.is_train) as scope:
            act = tf.cast(act, tf.float32)
            l = FC(act, 2048, 'act', initializer=tf.random_uniform_initializer(minval=-0.1, maxval=0.1))
        return l
    
    def _create_decoder(self, encode, act_embed):
        # encode: encode layer
        # act_embed: action embedding layer
        with tf.variable_scope('dec', reuse=not self.is_train) as scope:
            batch_size = tf.shape(encode)[0]
            l = tf.multiply(encode, act_embed, name='merge')
            l = FC(l, 1024, 'fc1')
            l = FC(l, 64 * 10 * 10, 'fc2')
            l = ReLu(l, 'relu1')
            l = tf.reshape(l, [-1, 10, 10, 64], name='dec-reshape')
            l = Deconv2D(l, [6, 6], [batch_size, 20, 20, 64], 64, 2, 'SAME', 'deconv1')
            l = ReLu(l, 'relu2')
            l = Deconv2D(l, [6, 6], [batch_size, 40, 40, 64], 64, 2, 'SAME', 'deconv2')
            l = ReLu(l, 'relu3')
            l = Deconv2D(l, [6, 6], [batch_size, 84, 84, NUM_CHANNELS], 3, 2, 'VALID', 'deconv3')
        return l
