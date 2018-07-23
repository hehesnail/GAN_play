import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter

def concat(tensors, axis, *args, **kwargs):
  return tf.concat(tensors, axis, *args, **kwargs)

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

class layer_norm(object):
  def __init__(self, name="layer_norm"):
    with tf.variable_scope(name):
      self.name = name
  def __call__(self, x):
    return tf.contrib.layers.layer_norm(x,
                    center=True,
                    scale=True,
                    trainable=True,
                    scope=self.name)

def l2_norm(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_norm(w, iteration=1):
  w_shape =  w.shape.as_list()
  w = tf.reshape(w, [-1, w_shape[-1]])

  u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
  u_hat = u
  v_hat = None
  for i in range(iteration):
    v_ = tf.matmul(u_hat, tf.transpose(w))
    v_hat = l2_norm(v_)

    u_ = tf.matmul(v_hat, w)
    u_hat = l2_norm(u_)

  sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
  w_norm = w / sigma

  with tf.control_dependencies([u.assign(u_hat)]):
    w_norm = tf.reshape(w_norm, w_shape)

  return w_norm


def hw_flatten(x):
  return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def attention_layer(input, name="attention", sn=False):
  """
  input: size [batch_size, H, W, C]
  f: [bs, H, W, C//8], g:[bs, H, W, C//8]
  s: [bs, N, N] N-> H x W
  h: [bs, H, W, C]

  out: size [batch_size, N, C]
  """
  with tf.variable_scope(name):
    #batch_size, h, w,
    c = input.get_shape()[-1]
    c_new = c // 8
    f = conv2d(input, c_new, k_h=1, k_w=1, d_h=1, d_w=1, name=name+'f', sn=sn)
    g = conv2d(input, c_new, k_h=1, k_w=1, d_h=1, d_w=1, name=name+'g', sn=sn)
    s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)

    #atten = tf.exp(s) / tf.reduce_sum(tf.exp(s), axis=-1, keep_dims=True)
    atten = tf.nn.softmax(s)
    h = conv2d(input, c, k_h=1, k_w=1, d_h=1, d_w=1, name=name+'h', sn=sn)

    o = tf.matmul(atten, hw_flatten(h))
    gamma = tf.get_variable('gamma', shape=[1], initializer=tf.constant_initializer(0.0))
    o = tf.reshape(o, shape=input.get_shape())
    input = gamma * o + input

    return input

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat([
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def conv2d(input_, output_dim,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d", sn=False):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    if sn:
      w = spectral_norm(w)
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", sn=False, with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))

    if sn:
      w = spectral_norm(w)
    deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv

def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, sn=False, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))

    if sn:
      matrix = spectral_norm(matrix)

    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias
