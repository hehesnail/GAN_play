from __future__ import division
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import pickle

from ops import *
from utils import *

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, sample_dir=None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    if not self.y_dim:
      self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir

    if self.dataset_name == 'cifar10':
      self.data_X = self.load_cifar10()
      self.c_dim = self.data_X[0].shape[-1]
    else:
      self.data = glob(os.path.join("../data", self.dataset_name, self.input_fname_pattern))
      imreadImg = imread(self.data[0])
      if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
        self.c_dim = imread(self.data[0]).shape[-1]
      else:
        self.c_dim = 1

    self.grayscale = (self.c_dim == 1)

    self.build_model()

  def build_model(self):
    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)

    self.G, self.g_att, self.g_gamma = self.generator(self.z, self.y)
    self.D, self.D_logits, self.d_att, self.d_gamma = self.discriminator(inputs, self.y, reuse=False)
    self.sampler, self.s_att, self.s_gamma = self.sampler(self.z, self.y)
    self.D_, self.D_logits_, _, _ = self.discriminator(self.G, self.y, reuse=True)

    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.g_gamma_sum = scalar_summary("g_gamma", tf.squeeze(self.g_gamma))
    self.d_gamma_sum = scalar_summary("d_gamma", tf.squeeze(self.d_gamma))


    self.d_loss_real = tf.reduce_mean(tf.nn.relu(1.0 - self.D_logits))
    self.d_loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + self.D_logits_))

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

    self.d_loss = self.d_loss_real + self.d_loss_fake

    self.g_loss = -tf.reduce_mean(self.D_logits_)

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    d_optim = tf.train.AdamOptimizer(0.0004, beta1=0.0, beta2=0.9) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(0.0001, beta1=0.0, beta2=0.9) \
              .minimize(self.g_loss, var_list=self.g_vars)

    #d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
    #          .minimize(self.d_loss, var_list=self.d_vars)
    #g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
    #          .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z_sum,
      self.d_loss_fake_sum, self.d__sum, self.g_loss_sum, self.g_gamma_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_loss_real_sum, self.d_sum, self.d_loss_sum, self.d_gamma_sum])
    self.writer = SummaryWriter("./logs", self.sess.graph)

    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))

    if config.dataset == 'cifar10':
      sample_inputs = self.data_X[0:self.sample_num]
      #sample_labels = self.data_y[0:self.sample_num]
    else:
      sample_files = self.data[0:self.sample_num]
      sample = [
          get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]
      if (self.grayscale):
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
      else:
        sample_inputs = np.array(sample).astype(np.float32)

    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in range(config.epoch):
      if config.dataset == 'cifar10':
        batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
      else:
        self.data = glob(os.path.join(
          "../data", config.dataset, self.input_fname_pattern))
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size

      for idx in range(0, batch_idxs):
        if config.dataset == 'cifar10':
          batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
        else:
          batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
          batch = [
              get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale) for batch_file in batch_files]
          if self.grayscale:
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
          else:
            batch_images = np.array(batch).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)

        if config.dataset == 'cifar10':
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={
              self.inputs: batch_images,
              self.z: batch_z,
            })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={
              self.z: batch_z,
            })
          self.writer.add_summary(summary_str, counter)
          #_, summary_str = self.sess.run([g_optim, self.g_sum],
          #  feed_dict={
          #    self.z: batch_z,
          #  })
          #self.writer.add_summary(summary_str, counter)

          errD_fake = self.d_loss_fake.eval({
              self.z: batch_z,
          })
          errD_real = self.d_loss_real.eval({
              self.inputs: batch_images,
          })
          errG = self.g_loss.eval({
              self.z: batch_z,
          })
        else:
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ self.inputs: batch_images, self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={ self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
          errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
          errG = self.g_loss.eval({self.z: batch_z})

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))

        if np.mod(counter, 100) == 1:
          if config.dataset == 'cifar10':
            samples, d_loss, g_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
              }
            )
            save_images(samples, image_manifold_size(samples.shape[0]),
                  './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
            print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
          else:
            try:
              samples, d_loss, g_loss = self.sess.run(
                [self.sampler, self.d_loss, self.g_loss],
                feed_dict={
                    self.z: sample_z,
                    self.inputs: sample_inputs,
                },
              )
              save_images(samples, image_manifold_size(samples.shape[0]),
                    './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
              print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
            except:
              print("one pic error!...")

        if np.mod(counter, 500) == 2:
          self.save(config.checkpoint_dir, counter)

  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      if not self.y_dim:
        # self.df_dim = 64
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv', sn=True))
        h1 = lrelu(conv2d(h0, self.df_dim*2, name='d_h1_conv', sn=True))
        h2 = lrelu(conv2d(h1, self.df_dim*4, name='d_h2_conv', sn=True))
        h2, atten, gamma = attention_layer(h2, name="d_att1", with_att=True)
        h3 = tf.layers.conv2d(h2, 1, (4,4), (1,1), padding='valid')
        h4 = tf.squeeze(h3)

        return tf.nn.sigmoid(h4), h4, atten, gamma

  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

        # project `z` and reshape
        z = tf.reshape(z, [self.batch_size, 1, 1, self.z_dim])
        h0 = deconv2d(z, [self.batch_size, s_h8, s_w8, self.gf_dim*4],
                k_h=4, k_w=4, d_h=4, d_w=4, name='g_h0', sn=True)
        h0 = tf.nn.relu(self.g_bn0(h0))

        h1= deconv2d(h0, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h1', sn=True)
        h1 = tf.nn.relu(self.g_bn1(h1))

        h2 = deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_h2', sn=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h2, atten, gamma = attention_layer(h2, name="g_att1", with_att=True)

        h3 = deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3')

        return tf.nn.tanh(h3), atten, gamma

  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()
      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

        # project `z` and reshape
        z = tf.reshape(z, [self.batch_size, 1, 1, self.z_dim])
        h0 = deconv2d(z, [self.batch_size, s_h8, s_w8, self.gf_dim*4],
                k_h=4, k_w=4, d_h=4, d_w=4, name='g_h0', sn=True)
        h0 = tf.nn.relu(self.g_bn0(h0))

        h1= deconv2d(h0, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h1', sn=True)
        h1 = tf.nn.relu(self.g_bn1(h1))

        h2 = deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim], name='g_h2', sn=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h2, atten, gamma = attention_layer(h2, name="g_att1", with_att=True)

        h3 = deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3')

        return tf.nn.tanh(h3), atten, gamma

  def load_cifar10(self):
    def load_cifar10_batch(filename):
      with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype(np.float)
      return X, Y
    xs = []
    ys = []
    for b in range(1,6):
      f = os.path.join("../data", self.dataset_name, 'data_batch_%d' % (b,))
      X, Y = load_cifar10_batch(f)
      xs.append(X)
      ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X
    del Y

    seed = 547
    np.random.seed(seed)
    np.random.shuffle(Xtr)
    np.random.seed(seed)
    np.random.shuffle(Ytr)

    return Xtr/127.5-1.0

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)

  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
