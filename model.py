import tensorflow as tf
import numpy as np

from tf_ops import Conv2D, FC, Deconv2D

NUM_CHANNELS = 3
NUM_FRAMES = 4

class ActionConditionalVideoPredictionModel(object):
    def __init__(self, inputs=None, optimizer_args=None):
        # optimizer_args: optimizer arguments (e.g. optimizer type, learning rate, ...) (dict)
        self.optimizer_args = optimizer_args
        self._create_input(inputs)
        self._create_model()
        self._create_output()
        self._create_loss()
        self._create_optimizer()

    def _create_input(self, inputs):
        if inputs == None:
            self.inputs = {'s_t': tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, (NUM_CHANNELS * NUM_FRAMES)]),
                       'a_t': tf.placeholder(dtype=tf.int32, shape=[None, 1]),
                       'x_t_1': tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, (NUM_CHANNELS)])}
        else:
            self.inputs = inputs
 
    def _create_model(self):
        self.encode = self._create_encoder(self.inputs['s_t'])
        self.act_embed = self._create_action_embedding(self.inputs['a_t'])
        self.decode = self._create_decoder(self.encode, self.act_embed)

    def _create_output(self):
        self.output = self.decode

    def _create_loss(self):  
        with tf.variable_scope('loss') as scope:
            t = self.inputs['x_t_1'] / 255.0
            self.loss = tf.reduce_mean(tf.nn.l2_loss(self.output - t, name='l2'))

    def _create_optimizer(self):
        with tf.variable_scope('train') as scope:
            lr = self.optimizer_args['lr']
            self.optimizer = tf.train.AdamOptimizer(learning_rate=lr, name='optimizer')
        self.train = self.optimizer.minimize(self.loss)

    def _create_encoder(self, x):
        # x: input image
        x = x / 255.0
        conv1 = Conv2D(x, [6, 6], 64, 2, 'VALID', 'conv1')
        conv2 = Conv2D(conv1, [6, 6], 64, 2, 'SAME', 'conv2')
        conv3 = Conv2D(conv2, [6, 6], 64, 2, 'SAME', 'conv3')
        fc1 = FC(conv3, 1024, 'enc-fc1')
        fc2 = FC(fc1, 2048, 'enc-fc2')
        return fc2

    def _create_action_embedding(self, act):
        act = tf.cast(act, tf.float32)
        fc1 = FC(act, 2048, 'act')
        return fc1
    
    def _create_decoder(self, encode, act_embed):
        # encode: encode layer
        # act_embed: action embedding layer
        batch_size = tf.shape(encode)[0]  
        merge = tf.multiply(encode, act_embed, name='merge')
        fc1 = FC(merge, 2048, 'dec-fc1')
        fc2 = FC(fc1, 1024, 'dec-fc2')
        dec = FC(fc2, 64 * 10 * 10, 'dec')
        dec = tf.reshape(dec, [-1, 10, 10, 64])
        deconv1 = Deconv2D(dec, [6, 6], [batch_size, 20, 20, 64], 64, 2, 'SAME', 'deconv1')
        deconv2 = Deconv2D(deconv1, [6, 6], [batch_size, 40, 40, 64], 64, 2, 'SAME', 'deconv2')
        deconv3 = Deconv2D(deconv2, [6, 6], [batch_size, 84, 84, NUM_CHANNELS], 3, 2, 'VALID', 'deconv3')
        return deconv3
