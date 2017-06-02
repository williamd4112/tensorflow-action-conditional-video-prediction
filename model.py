import tensorflow as tf
import numpy as np

NUM_CHANNELS = 3
NUM_FRAMES = 4

def Conv2D(x, filter_shape, out_dim, strides, padding, name):
    # x: input tensor (float32)[n, w, h, c]
    # filter_shape: conv2d filter (int)[w, h]
    # out_dim: output channels (int)
    # strides: conv2d stride size (int)
    # padding: padding type (str)
    # name: variable scope (str)
           
    with tf.variable_scope(name) as scope:
        in_dim = x.get_shape()[-1]
        w = tf.get_variable('w', shape=filter_shape + [in_dim, out_dim], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', shape=[out_dim], initializer=tf.constant_initializer(0.0))
        l = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=padding)
        l = tf.nn.bias_add(l, b)
        l = tf.nn.relu(l) 
    return l

def FC(x, out_dim, name):
    # x: input tensor (float32)[n, in_dim]
    # out_dim: output channels (int)
    # name: variable scope (str)

    x = tf.contrib.layers.flatten(x)
    with tf.variable_scope(name) as scope:
        in_dim = x.get_shape()[-1]
        w = tf.get_variable('w', shape=[in_dim, out_dim], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', shape=[out_dim], initializer=tf.constant_initializer(0.0))
        l = tf.add(tf.matmul(x, w), b)
        l = tf.nn.relu(l)
    return l

def Deconv2D(x, filter_shape, output_shape, out_dim, strides, padding, name):
    # x: input tensor (float32) [n, w, h, c]
    # filter_shape: conv2d filter (int)[w, h]
    # out_dim: output channels (int)
    # strides: conv2d stride size (int)
    # padding: padding type (str)
    # name: variable scope (str)
    with tf.variable_scope(name) as scope:
        in_dim = x.get_shape()[-1]
        w = tf.get_variable('w', shape=filter_shape + [out_dim, in_dim], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b', shape=[out_dim], initializer=tf.constant_initializer(0.0))
        l = tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, strides, strides, 1], padding=padding)
        l = tf.nn.bias_add(l, b)
        l = tf.nn.relu(l) 
    return l


class ActionConditionalVideoPredictionModel(object):
    def __init__(self):
        self._create_input()
        encode = self._create_encoder(self.inputs['x'])
        act_embed = self._create_action_embedding(self.inputs['act'])
        decode = self._create_decoder(encode, act_embed)

    def _create_input(self):
        self.inputs = {'x': tf.placeholder(dtype=tf.float32, shape=[None, 84, 84, (NUM_CHANNELS * NUM_FRAMES)]),
                       'act': tf.placeholder(dtype=tf.float32, shape=[None, 1])}
        return self.inputs

    def _create_encoder(self, x):
        # x: input image
        conv1 = Conv2D(x, [6, 6], 64, 2, 'VALID', 'conv1')
        conv2 = Conv2D(conv1, [6, 6], 64, 2, 'SAME', 'conv2')
        conv3 = Conv2D(conv2, [6, 6], 64, 2, 'SAME', 'conv3')
        fc1 = FC(conv3, 1024, 'enc-fc1')
        fc2 = FC(fc1, 2048, 'enc-fc2')
        return fc2

    def _create_action_embedding(self, act):
        fc1 = FC(act, 2048, 'act')
        return fc1
    
    def _create_decoder(self, encode, act_embed):
        # encode: encode layer
        # act_embed: action embedding layer
        merge = tf.multiply(encode, act_embed, name='merge')
        fc1 = FC(merge, 2048, 'dec-fc1')
        fc2 = FC(fc1, 1024, 'dec-fc2')
        dec = FC(fc2, 64 * 10 * 10, 'dec')
        dec = tf.reshape(dec, [-1, 10, 10, 64])
        deconv1 = Deconv2D(dec, [6, 6], [-1, 20, 20, 64], 64, 2, 'SAME', 'deconv1')
        deconv2 = Deconv2D(deconv1, [6, 6], [-1, 40, 40, 64], 64, 2, 'SAME', 'deconv2')
        deconv3 = Deconv2D(deconv2, [6, 6], [-1, 84, 84, NUM_CHANNELS], 3, 2, 'VALID', 'deconv3')
        return deconv3

if __name__ == '__main__':
    model = ActionConditionalVideoPredictionModel()
