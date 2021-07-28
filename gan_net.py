from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import logging
import json
import os
import gc
import datetime
import random
import h5py
import warnings
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

class HDFDataFrame:
    def __init__(self, file_name, batch_size):
        self.batch_size = batch_size
        self.file_name = file_name
        with h5py.File(self.file_name, 'r') as H:
            self.nrows = H['trainN'].shape[0]

    def get_data_batch(self, cols1, cols2):
        #r = sorted(np.random.randint(0,self.nrows,size=self.batch_size))
        r = sorted(random.sample([x for x in xrange(self.nrows)], self.batch_size))
        with h5py.File(self.file_name, 'r') as H:
            trainN = H['trainN'][r,:]
            trainG = H['trainG'][r,:]
            #colsN = H['trainN'].attrs['columns']
        return trainN, trainG

class ChunkedDataFrame:

    data_chunk     = None  # DataFrame to hold data chunk
    chunk_size     = None  # number of lines in the chunk
    chunk_n        = None  # chunk number
    chunk_iterator = iter([])

    data_batch     = None  # DataFrame to hold part of data_chunk
    batch_size     = None  # number of lines in the batch
    i              = None  # batch (draw) number
    max_i          = None  # max number of batches to draw from one chunk
    reuse_file     = None  # reuse file when its end is reached (false by default)
    logger         = None

    def __init__(self, file_name, chunk_size, batch_size, coverage, reuse_file = False, enable_log = False):
        self.check_arguments(file_name, chunk_size, batch_size, coverage, reuse_file)
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.file_name = file_name
        self.chunk_iterator = pd.read_csv(file_name, chunksize = self.chunk_size, header = 0)
        self.data_chunk = next(self.chunk_iterator)
        self.chunk_n = 0
        self.i = 0
        self.max_i = coverage * (chunk_size // batch_size) # how many times to draw random batch from one chunk
        self.coverage = coverage
        self.reuse_file = reuse_file
        self.init_logger(enable_log)
        print("-------------------------------------------")
        print("ChunkedDataFrame initialization parameters:")
        print("chunk_size: {0:,} lines".format(self.chunk_size))
        print("batch_size: {0:,} lines".format(self.batch_size))
        print("coverage:   {0:,}".format(self.coverage))
        print("reuse_file: {0}".format(self.reuse_file))
        print("-------------------------------------------")
        self.logger.info("Initialization parameters:")
        self.logger.info("chunk_size: {0:,} lines".format(self.chunk_size))
        self.logger.info("batch_size: {0:,} lines".format(self.batch_size))
        self.logger.info("coverage:   {0:,}".format(self.coverage))
        self.logger.info("reuse_file: {0}".format(self.reuse_file))
        self.logger.info("--------------------------".format(self.batch_size))
        self.logger.info("Loaded chunk {}".format(self.chunk_n+1))
        self.logger.info("Chunk size: {0:,} lines".format(self.data_chunk.shape[0]))
        self.logger.info("max_i: {0:,}".format(self.max_i))

    def init_logger(self, enable_log):
        logger = logging.getLogger('ChunkedDataFrame')
        logger.setLevel(logging.INFO)
        if enable_log:
            h = logging.FileHandler("data_reading_{0:%m%d-%H%M%S}.log".format(datetime.datetime.now()))
            formatter = logging.Formatter('%(asctime)s  %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
            h.setFormatter(formatter)
        else:
            h = logging.NullHandler()
        logger.addHandler(h)
        self.logger = logger

    def check_arguments(self, file_name, chunk_size, batch_size, coverage, reuse_file):
        try:
            df = pd.read_csv(file_name, header = None, nrows = 1)
        except IOError:
            print("ChunkedDataFrame: Can't open file {}".format(file_name))
            sys.exit(-1)
        assert(type(chunk_size) is int)
        assert(type(batch_size) is int)
        assert(chunk_size >= batch_size and chunk_size % batch_size == 0)
        assert(type(coverage) is int and coverage > 0)
        assert(type(reuse_file) is bool)

    def get_data_batch(self, cols_list):
        #self.logger.info("Drawing batch number {}".format(self.i+1))
        if self.i < self.max_i:
            ind = np.random.choice(self.data_chunk.shape[0], self.batch_size, replace = False)
            self.data_batch = self.data_chunk.iloc[ind, :]
            self.i = self.i + 1
        else:
            # load next chunk into memory
            try:
                self.data_chunk = next(self.chunk_iterator)
                self.chunk_n = self.chunk_n + 1
                self.logger.info("Loaded chunk {}".format(self.chunk_n+1))
                self.logger.info("Chunk size: {0:,} lines".format(self.data_chunk.shape[0]))

                # by the end of data file, chunk size may decrease so that max_i should be recalculated
                if self.data_chunk.shape[0] < self.batch_size:
                    raise StopIteration('Batch doesn\'t fit in the last chunk of the file')
                else: 
                    self.max_i = self.coverage * (self.data_chunk.shape[0] // self.batch_size)
                    self.logger.info("max_i: {0:,}".format(self.max_i))

            except StopIteration:
                if self.reuse_file:
                    self.logger.info("Reached the end of the file. Will start from the beginning.")
                    self.chunk_iterator = pd.read_csv(self.file_name, chunksize = self.chunk_size, header = 0)
                    self.data_chunk = next(self.chunk_iterator)
                    self.chunk_n = 0
                    self.logger.info("Loaded chunk {}".format(self.chunk_n+1))
                    self.logger.info("Chunk size: {}".format(self.data_chunk.shape[0]))
                    self.max_i = self.coverage * (self.chunk_size // self.batch_size)
                else:
                    self.logger.info("Reached the end of the file.")
                    self.data_chunk = None
                    self.data_batch = None
                    self.i = None
                    return self.data_batch

            self.i = 0
            self.get_data_batch(cols_list)
        if len(cols_list) == 2:
            return self.data_batch[cols_list[0]].values, self.data_batch[cols_list[1]].values
        else:
            return self.data_batch[cols_list[0]].values

class GAN():
    def __init__(self, header_file, json_file, model_folder, log_file, device):
        # assign necessary paths
        self.model_folder = model_folder
        self.header_file = header_file
        # read parameters
        self.__read_params__(json_file)
        # remember device
        self.device_name = device

    def __read_params__(self, json_file):
        with open(json_file, "r") as jfile:
            param = json.load(jfile)
        self.warm_start_iter = param['warm_start'] # 0 means no warm start!
        self.checkpoint_nm = 'check_point'
        self.num_iter = param['iter'] # num of iterations for learning
        self.mb_size = param['batch_size'] # mb_size = 128 #minibatch size..
        if 'num_gen_per_disc' in param.keys():
            self.num_gen_per_disc = param['num_gen_per_disc'] # num of times generator will be updated per 1 update of the discriminator
        else:
            self.num_gen_per_disc = 1
        if 'num_disc_per_gen' in param.keys():
            self.num_disc_per_gen = param['num_disc_per_gen'] # num of times generator will be updated per 1 update of the discriminator
        else:
            self.num_disc_per_gen = 1
        self.plot_ind = 0
        self.update_check_size = param['update_check_size'] # check loss function every .. iterations...
        self.save_model_iter = param['save_model_iter'] # save model every .. iterations...
        self.z_dim = param['generator_layers']['input']['size']
        self.lambd = param['lambd']  # multiplied with gradient penalty -- regularization parameter, use scale 1, 10, 100, ...
        self.G_layers = param['generator_layers']
        self.D_layers = param['discriminator_layers']
        self.G_optimizer = param['generator_optimizer']
        self.G_optimizer_param = param['generator_optimizer_param']
        self.D_optimizer = param['discriminator_optimizer']
        self.D_optimizer_param = param['discriminator_optimizer_param']

    def __init_logger__(self, log_file, gan_type):
        logger = logging.getLogger('train_tf_' + gan_type + '_logger')
        logger.setLevel(logging.INFO)
        ch = logging.FileHandler(log_file)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        self.logger = logger

    def __read_data__(self, train_file, cols=[]):
        if len(cols) > 0:
            data = pd.read_csv(train_file, nrows=2)
            ix = [i for i,x in enumerate(data.columns) if x in cols]
            data = pd.read_csv(train_file, usecols=ix)
        else:
            data = pd.read_csv(train_file)
        if False:
            self.stats = {}
            for col in data.columns:
                self.stats[col] = {}
                self.stats[col]['min'] = data[col].min()
                self.stats[col]['max'] = data[col].max()
            self.__save_json__(os.path.join(self.model_folder, 'stats.json'))
        return data.values

    def __load_json__(self, json_file):
        with open(json_file, "r") as jfile:
            self.stats = json.load(jfile)

    def __save_json__(self, json_file):
        with open(json_file, "w") as jfile:
            json.dump(self.stats, jfile, indent=4)

    def __xavier_init__(self, size):
        in_dim = size[0]
        xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
        return tf.random_normal(shape=size, stddev=xavier_stddev)

    def __init_weights__(self, layers, input_dim, scope='default', output_dim=0):
        prev_layer_name = 'input'
        prev_dim = input_dim
        while True:
            current_layers = [(layer_name, layer) for layer_name,layer in layers.items() if layers[layer_name]['previous'] == prev_layer_name]
            if len(current_layers)==0: # no more layers left...
                break
            elif len(current_layers) > 1: # several layers with the same predecessor...
                warning("Warning: several layers in generator have the same predecessor")
            layer = current_layers[0][1]
            layer_name = current_layers[0][0]
            if layer['type'] == 'dense':
                if layer_name == 'output':
                    if not 'size' in layer.keys():
                        cur_dim = output_dim
                    else:
                        cur_dim = layer['size']
                else:
                    cur_dim = layer['size']
                with tf.variable_scope(scope):
                    tf.get_variable(layer_name + '_W', initializer=self.__xavier_init__([prev_dim, cur_dim]))
                    tf.get_variable(layer_name + '_b', initializer=tf.zeros([cur_dim]))
                prev_dim = cur_dim
            prev_layer_name = layer_name

    def __forward_step__(self, layers, x, scope='default', is_training=True):
        output = x
        prev_layer_name = 'input'
        while True:
            current_layers = [(layer_name, layer) for layer_name,layer in layers.items() if layers[layer_name]['previous'] == prev_layer_name]
            if len(current_layers)==0: # no more layers left...
                break
            elif len(current_layers) > 1: # several layers with the same predecessor...
                warning("Warning: several layers in the network have the same predecessor")
            layer = current_layers[0][1]
            layer_name = current_layers[0][0]
            if layer['type'] == 'dense':
                with tf.variable_scope(scope, reuse=True):
                    W = tf.get_variable(layer_name + '_W')
                    b = tf.get_variable(layer_name + '_b')
                output = tf.matmul(output, W) + b
            if layer['type'] == 'relu':
                output = tf.nn.relu(output)
            if layer['type'] == 'sigmoid':
                output = tf.nn.sigmoid(output)
            if layer['type'] == 'dropout':
                output = tf.layers.dropout(output, rate=layer['rate'], training=is_training)
            prev_layer_name = layer_name
        return output

    def __init_optimizer__(self, optimizer, optimizer_param, cost, scope='default'):
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = scope)
        if optimizer == 'adam':
            if 'learning_rate' in optimizer_param.keys():
                lr = optimizer_param['learning_rate']
            if 'beta1' in optimizer_param.keys():
                beta1 = optimizer_param['beta1']
            else:
                beta1 = 0.9
            if 'beta2' in optimizer_param.keys():
                beta2 = optimizer_param['beta2']
            else:
                beta2 = 0.99
            if 'epsilon' in optimizer_param.keys():
                epsilon = optimizer_param['epsilon']
            else:
                epsilon = 1e-08
            train_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(cost, var_list=var_list),

        if optimizer == 'sgd':
            if 'learning_rate' in optimizer_param.keys():
                lr = optimizer_param['learning_rate']
            train_op = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost, var_list=var_list)

        if optimizer == 'momentum_sgd':
            if 'learning_rate' in optimizer_param.keys():
                lr = optimizer_param['learning_rate']
            if 'momentum' in optimizer_param.keys():
                momentum = optimizer_param['momentum']
            else:
                momentum = 0.0
            if 'nesterov' in optimizer_param.keys():
                if optimizer_param['nesterov'] == 1:
                    nesterov = True
                else:
                    nesterov = False
            else:
                nesterov = True
            train_op = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum, use_nesterov=nesterov).minimize(cost, var_list=var_list)

        if optimizer == 'rmsprop':
            if 'learning_rate' in optimizer_param.keys():
                lr = optimizer_param['learning_rate']
            if 'decay' in optimizer_param.keys():
                decay = optimizer_param['decay']
            else:
                decay = 0.9
            if 'momentum' in optimizer_param.keys():
                momentum = optimizer_param['momentum']
            else:
                momentum = 0.0
            train_op = tf.train.RMSPropOptimizer(learning_rate=lr, decay=decay, momentum=momentum).minimize(cost, var_list=var_list)

        return train_op

    def __init_metrics__(self):
        #a = tf.Variable([1.0, 0.0, 1.0, 0.0])
        #b = tf.Variable([0.2, 0.6, 0.8, 0.3])
        #self.disc_metric,_ = tf.metrics.auc(labels = a, predictions = b)

        self.disc_metric = tf.metrics.auc(labels = tf.concat(values=[tf.ones_like(self.D_real), tf.zeros_like(self.D_fake)], axis=0),
                                          predictions = tf.nn.sigmoid(tf.concat(values=[self.D_real, self.D_fake], axis=0)))

        self.gen_metric = tf.metrics.auc(labels = tf.ones_like(self.D_fake),
                                         predictions = tf.nn.sigmoid(self.D_fake))

    def sample_z(self, m, n):
        return np.random.normal(0.0, 1.0, [m, n])

    def next_batch(self, data_size, batch_size):
        return np.random.choice(data_size, batch_size, replace=False)

    def get_perturbed_batch(self, minibatch):
        return minibatch + 0.5 * minibatch.std() * np.random.random(minibatch.shape)

    def sigmoid(x):
        s = 1/(1+np.exp(-x))
        return s

    def plot_errors(self, plotG, plotD, it_vec):
        if self.plot_ind == 1:
            # plot is always about original full loss vector...
            plt.plot(it_vec, np.array(plotD), label="Discriminator")
            plt.plot(it_vec, np.array(plotG), label="Generator")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(os.path.join(self.model_folder, 'G_D_loss.png'))
            plt.clf()

class DRAGAN(GAN):
    def __init__(self, header_file, json_file, model_folder, log_file, device):
        GAN.__init__(self, header_file, json_file, model_folder, log_file, device)
        # inititalize logger file
        self.__init_logger__(log_file, 'dragan')
        header = pd.read_csv(header_file)
        ix = [i for i,x in enumerate(header.iloc[0]) if x in ['N','G']]
        self.colnames = header.columns[ix]
        self.X_dim = len(self.colnames)

    def costs(self):
        self.D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(self.D_real)))
        self.D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
        self.disc_cost = self.D_loss_real + self.D_loss_fake

        # define loss function for generator here because it is based on discriminator
        self.gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))

        # Gradient penalty
        self.alpha = tf.random_uniform(
            shape=[self.mb_size, 1],
            minval=0.,
            maxval=1.
        )
        self.differences = self.X_p - self.X
        self.interpolates = self.X + (self.alpha * self.differences)
        self.gradients = tf.gradients(self.__forward_step__(self.D_layers, self.interpolates, scope='D'), [self.interpolates])[0]
        self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), reduction_indices=[1]))
        self.gradient_penalty = tf.reduce_mean((self.slopes - 1.) ** 2)
        self.disc_cost += self.lambd * self.gradient_penalty

        # define optimizers
        self.gen_train_op = self.__init_optimizer__(self.G_optimizer, self.G_optimizer_param, self.gen_cost, scope='G')
        self.disc_train_op = self.__init_optimizer__(self.D_optimizer, self.D_optimizer_param, self.disc_cost, scope='D')

    def generator(self, is_training=True):
        # z is a placeholder, which is the sampled data....
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.__init_weights__(self.G_layers, self.z_dim, output_dim=self.X_dim, scope='G')
        self.G_sample = self.__forward_step__(self.G_layers, self.z, scope='G')

    def discriminator(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim])
        self.X_p = tf.placeholder(tf.float32, shape=[None, self.X_dim])
        self.__init_weights__(self.D_layers, self.X_dim, scope='D')

        self.D_real = self.__forward_step__(self.D_layers, self.X, scope='D')
        self.D_fake = self.__forward_step__(self.D_layers, self.G_sample, scope='D')

    def fit(self, train_file):
        print("DRAGAN training started")
        print("Number of iterations to run: {0:,}".format(self.num_iter))
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
            self.warm_start_iter = 0
        data = self.__read_data__(train_file, self.colnames)
        data_size = data.shape[0]
        with tf.device(self.device_name):
            self.generator(is_training=True)
            self.discriminator()
            self.costs()
        self.saver = tf.train.Saver(max_to_keep=10000)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if self.warm_start_iter != 0:
            ckpt_nm_start = os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(self.warm_start_iter) + '.ckpt')
            self.saver.restore(self.sess, ckpt_nm_start)
            print("Warm start is used, warm_start_iter = {0:,}".format(self.warm_start_iter))

        plotD = []
        plotG = []
        it_vec = []
        start_index = 0
        end_index = 0
        for it in range(1 + self.warm_start_iter, self.num_iter + self.warm_start_iter + 1):
            end_index += 1
            for _ in range(self.num_gen_per_disc):
                # training of generator....
                _, G_loss_curr = self.sess.run(
                    [self.gen_train_op, self.gen_cost],
                    feed_dict={self.z: self.sample_z(self.mb_size, self.z_dim)}
                )

            X_mb = data[self.next_batch(data_size, self.mb_size),:]
            X_mb_p = self.get_perturbed_batch(X_mb)

            for _ in range(self.num_disc_per_gen):
                _, D_loss_curr, penalty = self.sess.run(
                    [self.disc_train_op, self.disc_cost, self.gradient_penalty],
                    feed_dict={self.X: X_mb, self.X_p: X_mb_p, self.z: self.sample_z(self.mb_size, self.z_dim)}
                )

            # record the loss for every iteration!
            plotD.append(D_loss_curr)
            plotG.append(G_loss_curr)
            it_vec.append(it)
            if it % self.save_model_iter == 0:
                # save the model!
                self.saver.save(self.sess, os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(it) + '.ckpt'))
            if it % self.update_check_size == 0:
                # every certain learning steps, have an update check......
                if self.plot_ind == 1:
                    self.plot_errors(plotG, plotD, it_vec)
                self.logger.info('Iter: {}; D loss: {:.4}; G loss: {:.4}; penalty: {:.4}'.format(it, np.mean(plotD[start_index: end_index]), np.mean(plotG[start_index: end_index]), penalty))
                start_index = end_index

    def generate(self, output_file, num_sample_sim, using_iter):
        with tf.device(self.device_name):
            self.generator(is_training=False)
        self.saver = tf.train.Saver(max_to_keep=10000)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        #read the model and sample the data!!!!
        if not os.path.exists(self.model_folder):
            warning("Warning: there are no check points in model folder for data generation")
        else:
            ckpt_nm = os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(using_iter) + '.ckpt')
            self.saver.restore(self.sess, ckpt_nm)
        G_sample = self.sess.run(
            self.G_sample,
            feed_dict={self.z: self.sample_z(num_sample_sim, self.z_dim)}
        )
        G_sample_df = pd.DataFrame(G_sample, columns=self.colnames)
        G_sample_df.reset_index(drop=True, inplace=True)
        G_sample_df.to_csv(output_file, index=False)

class CycleGAN(GAN):
    def __init__(self, header_file_A, header_file_B, json_file, model_folder, log_file, device):
        # assign necessary paths
        self.model_folder = model_folder
        self.header_file_A = header_file_A
        self.header_file_B = header_file_B
        header = pd.read_csv(header_file_A)
        ix = [i for i,x in enumerate(header.iloc[0]) if x in ['N','G']]
        self.colnames_A = header.columns[ix]
        self.A_dim = len(self.colnames_A)
        header = pd.read_csv(header_file_B)
        ix = [i for i,x in enumerate(header.iloc[0]) if x in ['N','G']]
        self.colnames_B = header.columns[ix]
        self.B_dim = len(self.colnames_B)
        # read parameters
        self.__read_params__(json_file)
        # remember device
        self.device_name = device
        # inititalize logger file
        self.__init_logger__(log_file, 'cyclegan')

    def __read_params__(self, json_file):
        with open(json_file, "r") as jfile:
            param = json.load(jfile)
        self.warm_start_iter = param['warm_start'] # 0 means no warm start!
        self.checkpoint_nm = 'check_point'
        self.num_iter = param['iter'] # num of iterations for learning
        self.mb_size = param['batch_size'] # mb_size = 128 #minibatch size..
        if 'num_gen_per_disc' in param.keys():
            self.num_gen_per_disc = param['num_gen_per_disc'] # num of times generator will be updated per 1 update of the discriminator
        else:
            self.num_gen_per_disc = 1
        if 'num_disc_per_gen' in param.keys():
            self.num_disc_per_gen = param['num_disc_per_gen'] # num of times generator will be updated per 1 update of the discriminator
        else:
            self.num_disc_per_gen = 1
        self.plot_ind = 0
        self.cycle_lambd = param['cycle_lambd']
        self.update_check_size = param['update_check_size'] # check loss function every .. iterations...
        self.save_model_iter = param['save_model_iter'] # save model every .. iterations...
        self.G_layers = param['generator_layers']
        self.D_layers = param['discriminator_layers']
        self.G_optimizer = param['generator_optimizer']
        self.G_optimizer_param = param['generator_optimizer_param']
        self.D_optimizer = param['discriminator_optimizer']
        self.D_optimizer_param = param['discriminator_optimizer_param']
        if 'loss' in param.keys():
            self.loss = param['loss']
        else:
            self.loss = 'mse'

    def generator_AtoB(self, is_training=True):
        self.input_A = tf.placeholder(tf.float32, shape=[None, self.A_dim])
        self.__init_weights__(self.G_layers, self.A_dim, output_dim=self.B_dim, scope='gen_AtoB')
        self.gen_B = self.__forward_step__(self.G_layers, self.input_A, scope='gen_AtoB')

    def generator_BtoA(self, is_training=True):
        self.input_B = tf.placeholder(tf.float32, shape=[None, self.B_dim])
        self.__init_weights__(self.G_layers, self.B_dim, output_dim=self.A_dim, scope='gen_BtoA')
        self.gen_A = self.__forward_step__(self.G_layers, self.input_B, scope='gen_BtoA')

    def discriminator_A(self):
        self.__init_weights__(self.D_layers, self.A_dim, scope='disc_A')
        self.dec_A = self.__forward_step__(self.D_layers, self.input_A, scope='disc_A')
        self.dec_gen_A = self.__forward_step__(self.D_layers, self.gen_A, scope='disc_A')

    def discriminator_B(self):
        self.__init_weights__(self.D_layers, self.B_dim, scope='disc_B')
        self.dec_B = self.__forward_step__(self.D_layers, self.input_B, scope='disc_B')
        self.dec_gen_B = self.__forward_step__(self.D_layers, self.gen_B, scope='disc_B')

    def cycles(self):
        self.cyc_A = self.__forward_step__(self.G_layers, self.gen_B, scope='gen_BtoA')
        self.cyc_B = self.__forward_step__(self.G_layers, self.gen_A, scope='gen_AtoB')

    def costs(self):
        cyc_loss = tf.reduce_mean(tf.abs(self.input_A-self.cyc_A)) + tf.reduce_mean(tf.abs(self.input_B-self.cyc_B))
        self.identity_A = self.__forward_step__(self.G_layers, self.input_A, scope='gen_BtoA')
        self.identity_B = self.__forward_step__(self.G_layers, self.input_B, scope='gen_AtoB')
        identity_loss = tf.reduce_mean(tf.abs(self.input_A-self.identity_A)) + tf.reduce_mean(tf.abs(self.input_B-self.identity_B))

        if self.loss == 'mse':
            self.gen_AtoB_loss = tf.reduce_mean(tf.squared_difference(self.dec_gen_B,1)) + self.cycle_lambd*cyc_loss + identity_loss
            self.gen_BtoA_loss = tf.reduce_mean(tf.squared_difference(self.dec_gen_A,1)) + self.cycle_lambd*cyc_loss + identity_loss

            self.disc_A_loss = (tf.reduce_mean(tf.squared_difference(self.dec_A,1)) + tf.reduce_mean(tf.square(self.dec_gen_A)))/2.0
            self.disc_B_loss = (tf.reduce_mean(tf.squared_difference(self.dec_B,1)) + tf.reduce_mean(tf.square(self.dec_gen_B)))/2.0

        if self.loss == 'logloss':
            self.gen_AtoB_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dec_gen_B, labels=tf.ones_like(self.dec_gen_B))) + self.cycle_lambd*cyc_loss + identity_loss
            self.gen_BtoA_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dec_gen_A, labels=tf.ones_like(self.dec_gen_A))) + self.cycle_lambd*cyc_loss + identity_loss

            disc_A_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dec_A, labels=tf.ones_like(self.dec_A)))
            disc_A_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dec_gen_A, labels=tf.zeros_like(self.dec_gen_A)))
            self.disc_A_loss = (disc_A_loss_real + disc_A_loss_fake)/2.0

            disc_B_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dec_B, labels=tf.ones_like(self.dec_B)))
            disc_B_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dec_gen_B, labels=tf.zeros_like(self.dec_gen_B)))
            self.disc_B_loss = (disc_B_loss_real + disc_B_loss_fake)/2.0

        # define optimizers
        self.gen_AtoB_train_op = self.__init_optimizer__(self.G_optimizer, self.G_optimizer_param, self.gen_AtoB_loss, scope='gen_AtoB')
        self.gen_BtoA_train_op = self.__init_optimizer__(self.G_optimizer, self.G_optimizer_param, self.gen_BtoA_loss, scope='gen_BtoA')
        self.disc_A_train_op = self.__init_optimizer__(self.D_optimizer, self.D_optimizer_param, self.disc_A_loss, scope='disc_A')
        self.disc_B_train_op = self.__init_optimizer__(self.D_optimizer, self.D_optimizer_param, self.disc_B_loss, scope='disc_B')

    def fit(self, train_file_A, train_file_B):
        print("CycleGAN training started")
        print("Number of iterations to run: {0:,}".format(self.num_iter))
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
            self.warm_start_iter = 0
        train_file_ext = os.path.basename(train_file_A).split('.')[-1]
        if train_file_ext == 'csv':
            data_source_A = ChunkedDataFrame(train_file_A, chunk_size = 500*self.mb_size, batch_size = self.mb_size, coverage = 10, reuse_file = True)
        elif train_file_ext == 'h5':
            data_source_A = HDFDataFrame(train_file_A, self.mb_size)
        train_file_ext = os.path.basename(train_file_B).split('.')[-1]
        if train_file_ext == 'csv':
            data_source_B = ChunkedDataFrame(train_file_B, chunk_size = 500*self.mb_size, batch_size = self.mb_size, coverage = 10, reuse_file = True)
        elif train_file_ext == 'h5':
            data_source_B = HDFDataFrame(train_file_B, self.mb_size)
        with tf.device(self.device_name):
            self.generator_AtoB(is_training=True)
            self.generator_BtoA(is_training=True)
            self.discriminator_A()
            self.discriminator_B()
            self.cycles()
            self.costs()
        self.saver = tf.train.Saver(max_to_keep=10000)
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init)
        if self.warm_start_iter != 0:
            ckpt_nm_start = os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(self.warm_start_iter) + '.ckpt')
            self.saver.restore(self.sess, ckpt_nm_start)
            print("Warm start is used, warm_start_iter = {0:,}".format(self.warm_start_iter))

        plot_genAtoB = []
        plot_genBtoA = []
        plot_discA = []
        plot_discB = []
        it_vec = []
        start_index = 0
        end_index = 0
        for it in range(1 + self.warm_start_iter, self.num_iter + self.warm_start_iter + 1):
            end_index += 1
            A_mb = data_source_A.get_data_batch([self.colnames_A])
            B_mb = data_source_B.get_data_batch([self.colnames_B])

            for _ in range(self.num_gen_per_disc):
                # training of generator....
                _, _, gen_AtoB_loss_curr, gen_BtoA_loss_curr, gen_B_temp, gen_A_temp = self.sess.run(
                    [self.gen_AtoB_train_op,
                     self.gen_BtoA_train_op,
                     self.gen_AtoB_loss,
                     self.gen_BtoA_loss,
                     self.gen_B,
                     self.gen_A],
                    feed_dict={self.input_A: A_mb, self.input_B: B_mb}
                )

            for _ in range(self.num_disc_per_gen):
                _, _, disc_A_loss_curr, disc_B_loss_curr = self.sess.run(
                    [self.disc_A_train_op,
                     self.disc_B_train_op,
                     self.disc_A_loss,
                     self.disc_B_loss],
                    feed_dict={self.input_A: A_mb,
                               self.input_B: B_mb,
                               self.gen_A: gen_A_temp,
                               self.gen_B: gen_B_temp}
                )

            # record the loss for every iteration!
            plot_genAtoB.append(gen_AtoB_loss_curr)
            plot_genBtoA.append(gen_BtoA_loss_curr)
            plot_discA.append(disc_A_loss_curr)
            plot_discB.append(disc_B_loss_curr)
            it_vec.append(it)

            if it % self.save_model_iter == 0:
                # save the model!
                self.saver.save(self.sess, os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(it) + '.ckpt'))

            if it % self.update_check_size == 0:
                # every certain learning steps, have an update check......
                if self.plot_ind == 1:
                    self.plot_errors(plotG, plotD, it_vec)
                loss_string = 'Iter: {}; Gen AtoB loss: {:.4}; Gen BtoA loss: {:.4}; Disc A loss: {:.4}; Disc B loss: {:.4}'
                self.logger.info(loss_string.format(it,
                                                    np.mean(plot_genAtoB[start_index: end_index]),
                                                    np.mean(plot_genBtoA[start_index: end_index]),
                                                    np.mean(plot_discA[start_index:end_index]),
                                                    np.mean(plot_discB[start_index:end_index])))
                start_index = end_index

    def generate(self, train_file, output_file, num_sample_sim, using_iter, how='as_train'):
        '''
        if how == 'as_train':
            conditions = self.__read_data__(train_file, self.conditions_colnames)
        if how == 'uniform':
            conditions = self.__read_data__(train_file, self.conditions_colnames)
            conditions = np.vstack({tuple(row) for row in conditions})
            #self.conditions = np.unique(self.conditions, axis=0)
        row_nm = np.random.choice(conditions.shape[0], num_sample_sim, replace=True)
        conditions = conditions[row_nm,:]
        '''
        with tf.device(self.device_name):
            self.generator_AtoB(is_training=True)
        self.saver = tf.train.Saver(max_to_keep=10000)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        #read the model and sample the data
        if not os.path.exists(self.model_folder):
            warning("Warning: there are no check points in model folder for data generation")
        else:
            ckpt_nm = os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(using_iter) + '.ckpt')
            self.saver.restore(self.sess, ckpt_nm)
        data_iterator = pd.read_csv(train_file, chunksize = 500000)
        data_batch = next(data_iterator)
        batchsize = 5000

        start_index = 0
        append = False
        sample_size = 0
        while sample_size < num_sample_sim:
            if start_index < len(data_batch):
                end_index = min(start_index + batchsize, len(data_batch))
                A_mb = data_batch.iloc[start_index:end_index][self.colnames_A].values
                start_index = end_index
            else:
                data_batch = next(data_iterator)
                start_index = 0
                end_index = min(start_index + batchsize, len(data_batch))
                A_mb = data_batch.iloc[start_index:end_index][self.colnames_A].values
                start_index = end_index
            gen_B = self.sess.run(
                self.gen_B,
                feed_dict={self.input_A: A_mb}
            )
            G_sample_batch_df = pd.DataFrame(gen_B, columns=self.colnames_B)
            sample_size += len(G_sample_batch_df)
            if not append:
                G_sample_batch_df.to_csv(output_file, index=False)
                append = True
            else:
                G_sample_batch_df.to_csv(output_file, mode='a', index=False, header=False)
            gc.collect()

class CycleDRAGAN(GAN):
    def __init__(self, header_file_A, header_file_B, json_file, model_folder, log_file, device):
        # assign necessary paths
        self.model_folder = model_folder
        self.header_file_A = header_file_A
        self.header_file_B = header_file_B
        header = pd.read_csv(header_file_A)
        ix = [i for i,x in enumerate(header.iloc[0]) if x in ['N','G']]
        self.colnames_A = header.columns[ix]
        self.A_dim = len(self.colnames_A)
        header = pd.read_csv(header_file_B)
        ix = [i for i,x in enumerate(header.iloc[0]) if x in ['N','G']]
        self.colnames_B = header.columns[ix]
        self.B_dim = len(self.colnames_B)
        # read parameters
        self.__read_params__(json_file)
        # remember device
        self.device_name = device
        # inititalize logger file
        self.__init_logger__(log_file, 'cycledragan')

    def __read_params__(self, json_file):
        with open(json_file, "r") as jfile:
            param = json.load(jfile)
        self.warm_start_iter = param['warm_start'] # 0 means no warm start!
        self.checkpoint_nm = 'check_point'
        self.num_iter = param['iter'] # num of iterations for learning
        self.mb_size = param['batch_size'] # mb_size = 128 #minibatch size..
        if 'num_gen_per_disc' in param.keys():
            self.num_gen_per_disc = param['num_gen_per_disc'] # num of times generator will be updated per 1 update of the discriminator
        else:
            self.num_gen_per_disc = 1
        if 'num_disc_per_gen' in param.keys():
            self.num_disc_per_gen = param['num_disc_per_gen'] # num of times generator will be updated per 1 update of the discriminator
        else:
            self.num_disc_per_gen = 1
        self.plot_ind = 0
        self.lambd = param['lambd']  # multiplied with gradient penalty -- regularization parameter, use scale 1, 10, 100, ...
        self.cycle_lambd = param['cycle_lambd']
        self.update_check_size = param['update_check_size'] # check loss function every .. iterations...
        self.save_model_iter = param['save_model_iter'] # save model every .. iterations...
        self.G_layers = param['generator_layers']
        self.D_layers = param['discriminator_layers']
        self.G_optimizer = param['generator_optimizer']
        self.G_optimizer_param = param['generator_optimizer_param']
        self.D_optimizer = param['discriminator_optimizer']
        self.D_optimizer_param = param['discriminator_optimizer_param']

    def generator_AtoB(self, is_training=True):
        self.input_A = tf.placeholder(tf.float32, shape=[None, self.A_dim])
        self.input_A_p = tf.placeholder(tf.float32, shape=[None, self.A_dim])
        self.__init_weights__(self.G_layers, self.A_dim, output_dim=self.B_dim, scope='gen_AtoB')
        self.gen_B = self.__forward_step__(self.G_layers, self.input_A, scope='gen_AtoB')
        self.identity_A = self.__forward_step__(self.G_layers, self.input_A, scope='gen_BtoA')

    def generator_BtoA(self, is_training=True):
        self.input_B = tf.placeholder(tf.float32, shape=[None, self.B_dim])
        self.input_B_p = tf.placeholder(tf.float32, shape=[None, self.B_dim])
        self.__init_weights__(self.G_layers, self.B_dim, output_dim=self.A_dim, scope='gen_BtoA')
        self.gen_A = self.__forward_step__(self.G_layers, self.input_B, scope='gen_BtoA')
        self.identity_B = self.__forward_step__(self.G_layers, self.input_B, scope='gen_AtoB')

    def discriminator_A(self):
        self.__init_weights__(self.D_layers, self.A_dim, scope='disc_A')
        self.dec_A = self.__forward_step__(self.D_layers, self.input_A, scope='disc_A')
        self.dec_gen_A = self.__forward_step__(self.D_layers, self.gen_A, scope='disc_A')

    def discriminator_B(self):
        self.__init_weights__(self.D_layers, self.B_dim, scope='disc_B')
        self.dec_B = self.__forward_step__(self.D_layers, self.input_B, scope='disc_B')
        self.dec_gen_B = self.__forward_step__(self.D_layers, self.gen_B, scope='disc_B')

    def cycles(self):
        self.cyc_A = self.__forward_step__(self.G_layers, self.gen_B, scope='gen_BtoA')
        self.cyc_B = self.__forward_step__(self.G_layers, self.gen_A, scope='gen_AtoB')

    def costs(self):
        cyc_loss = tf.reduce_mean(tf.abs(self.input_A-self.cyc_A)) + tf.reduce_mean(tf.abs(self.input_B-self.cyc_B))
        identity_loss = tf.reduce_mean(tf.abs(self.input_A-self.identity_A)) + tf.reduce_mean(tf.abs(self.input_B-self.identity_B))

        # version 1
        self.gen_AtoB_loss = tf.reduce_mean(tf.squared_difference(self.dec_gen_B,1)) + self.cycle_lambd*cyc_loss + identity_loss
        self.gen_BtoA_loss = tf.reduce_mean(tf.squared_difference(self.dec_gen_A,1)) + self.cycle_lambd*cyc_loss + identity_loss

        '''
        # version 2
        self.gen_AtoB_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dec_gen_B, labels=tf.ones_like(self.dec_gen_B))) + self.cycle_lambd*cyc_loss
        self.gen_BtoA_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dec_gen_A, labels=tf.ones_like(self.dec_gen_A))) + self.cycle_lambd*cyc_loss
        '''

        # version 1
        self.disc_A_loss = (tf.reduce_mean(tf.squared_difference(self.dec_A,1)) + tf.reduce_mean(tf.square(self.dec_gen_A)))/2.0
        self.disc_B_loss = (tf.reduce_mean(tf.squared_difference(self.dec_B,1)) + tf.reduce_mean(tf.square(self.dec_gen_B)))/2.0

        '''
        # version 2
        disc_A_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dec_A, labels=tf.ones_like(self.dec_A)))
        disc_A_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dec_gen_A, labels=tf.zeros_like(self.dec_gen_A)))
        self.disc_A_loss = (disc_A_loss_real + disc_A_loss_fake)/2.0

        disc_B_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dec_B, labels=tf.ones_like(self.dec_B)))
        disc_B_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.dec_gen_B, labels=tf.zeros_like(self.dec_gen_B)))
        self.disc_B_loss = (disc_B_loss_real + disc_B_loss_fake)/2.0
        '''

        # Gradient penalty
        self.alpha = tf.random_uniform(
            shape=[self.mb_size, 1],
            minval=0.,
            maxval=1.
        )
        self.differences = self.input_A_p - self.input_A
        self.A_interpolates = self.input_A + (self.alpha * self.differences)
        self.gradients = tf.gradients(self.__forward_step__(self.D_layers, self.A_interpolates, scope='disc_A'), [self.A_interpolates])[0]
        self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), reduction_indices=[1]))
        self.disc_A_gradient_penalty = tf.reduce_mean((self.slopes - 1.) ** 2)
        self.disc_A_loss += self.lambd * self.disc_A_gradient_penalty

        self.differences = self.input_B_p - self.input_B
        self.B_interpolates = self.input_B + (self.alpha * self.differences)
        self.gradients = tf.gradients(self.__forward_step__(self.D_layers, self.B_interpolates, scope='disc_B'), [self.B_interpolates])[0]
        self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), reduction_indices=[1]))
        self.disc_B_gradient_penalty = tf.reduce_mean((self.slopes - 1.) ** 2)
        self.disc_B_loss += self.lambd * self.disc_B_gradient_penalty

        # define optimizers
        self.gen_AtoB_train_op = self.__init_optimizer__(self.G_optimizer, self.G_optimizer_param, self.gen_AtoB_loss, scope='gen_AtoB')
        self.gen_BtoA_train_op = self.__init_optimizer__(self.G_optimizer, self.G_optimizer_param, self.gen_BtoA_loss, scope='gen_BtoA')
        self.disc_A_train_op = self.__init_optimizer__(self.D_optimizer, self.D_optimizer_param, self.disc_A_loss, scope='disc_A')
        self.disc_B_train_op = self.__init_optimizer__(self.D_optimizer, self.D_optimizer_param, self.disc_B_loss, scope='disc_B')

        #var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'disc_B')
        #self.grads_disc_B_loss = tf.gradients(self.disc_B_loss, var_list)
        #for index, grad in enumerate(grads):
        #    tf.summary.histogram("{}-grad".format(grads[index][1].name), grads[index])
        #self.summaries = tf.summary.merge_all()

    def fit(self, train_file_A, train_file_B, viz_folder='/axp/rim/skytreeml/dev/defimov/gan/fraud/cycledragan/viz_hist'):
        print("CycleDRAGAN training started")
        print("Number of iterations to run: {0:,}".format(self.num_iter))
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
            self.warm_start_iter = 0
        train_file_ext = os.path.basename(train_file_A).split('.')[-1]
        if train_file_ext == 'csv':
            data_source_A = ChunkedDataFrame(train_file_A, chunk_size = 500*self.mb_size, batch_size = self.mb_size, coverage = 10, reuse_file = True)
        elif train_file_ext == 'h5':
            data_source_A = HDFDataFrame(train_file_A, self.mb_size)
        train_file_ext = os.path.basename(train_file_B).split('.')[-1]
        if train_file_ext == 'csv':
            data_source_B = ChunkedDataFrame(train_file_B, chunk_size = 500*self.mb_size, batch_size = self.mb_size, coverage = 10, reuse_file = True)
        elif train_file_ext == 'h5':
            data_source_B = HDFDataFrame(train_file_B, self.mb_size)
        with tf.device(self.device_name):
            self.generator_AtoB(is_training=True)
            self.generator_BtoA(is_training=True)
            self.discriminator_A()
            self.discriminator_B()
            self.cycles()
            self.costs()
        self.saver = tf.train.Saver(max_to_keep=10000)
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init)
        if self.warm_start_iter != 0:
            ckpt_nm_start = os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(self.warm_start_iter) + '.ckpt')
            self.saver.restore(self.sess, ckpt_nm_start)
            print("Warm start is used, warm_start_iter = {0:,}".format(self.warm_start_iter))

        plot_genAtoB = []
        plot_genBtoA = []
        plot_discA = []
        plot_discB = []
        it_vec = []
        start_index = 0
        end_index = 0
        for it in range(1 + self.warm_start_iter, self.num_iter + self.warm_start_iter + 1):
            end_index += 1
            A_mb = data_source_A.get_data_batch([self.colnames_A])
            B_mb = data_source_B.get_data_batch([self.colnames_B])
            A_mb_p = self.get_perturbed_batch(A_mb)
            B_mb_p = self.get_perturbed_batch(B_mb)

            for _ in range(self.num_gen_per_disc):
                # training of generator....
                _, _, gen_AtoB_loss_curr, gen_BtoA_loss_curr, gen_B_temp, gen_A_temp = self.sess.run(
                    [self.gen_AtoB_train_op,
                     self.gen_BtoA_train_op,
                     self.gen_AtoB_loss,
                     self.gen_BtoA_loss,
                     self.gen_B,
                     self.gen_A],
                    feed_dict={self.input_A: A_mb, self.input_B: B_mb}
                )

            for _ in range(self.num_disc_per_gen):
                _, _, disc_B_loss_curr, disc_A_loss_curr, disc_B_penalty, disc_A_penalty = self.sess.run(
                    [self.disc_B_train_op,
                     self.disc_A_train_op,
                     self.disc_B_loss,
                     self.disc_A_loss,
                     self.disc_B_gradient_penalty,
                     self.disc_A_gradient_penalty],
                    feed_dict={self.input_A: A_mb,
                               self.input_B: B_mb,
                               self.input_A_p: A_mb_p,
                               self.input_B_p: B_mb_p,
                               self.gen_A: gen_A_temp,
                               self.gen_B: gen_B_temp}
                )
            #self.logger.info([np.max(x) for x in grads])
            #self.logger.info([np.min(x) for x in grads])

            # record the loss for every iteration!
            plot_genAtoB.append(gen_AtoB_loss_curr)
            plot_genBtoA.append(gen_BtoA_loss_curr)
            plot_discA.append(disc_A_loss_curr)
            plot_discB.append(disc_B_loss_curr)
            it_vec.append(it)

            if it % self.save_model_iter == 0:
                # save the model!
                self.saver.save(self.sess, os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(it) + '.ckpt'))

            if it % self.update_check_size == 0:
                # every certain learning steps, have an update check......
                if self.plot_ind == 1:
                    self.plot_errors(plotG, plotD, it_vec)
                loss_string = 'Iter: {}; Gen AtoB loss: {:.4}; Gen BtoA loss: {:.4}; Disc A loss: {:.4}; Disc B loss: {:.4}; Disc A penalty: {:.5}; Disc B penalty: {:.5}'
                self.logger.info(loss_string.format(it,
                                                    np.mean(plot_genAtoB[start_index: end_index]),
                                                    np.mean(plot_genBtoA[start_index: end_index]),
                                                    np.mean(plot_discA[start_index:end_index]),
                                                    np.mean(plot_discB[start_index:end_index]),
                                                    disc_A_penalty,
                                                    disc_B_penalty))
                start_index = end_index

    def generate(self, train_file, output_file, num_sample_sim, using_iter, how='as_train'):
        '''
        if how == 'as_train':
            conditions = self.__read_data__(train_file, self.conditions_colnames)
        if how == 'uniform':
            conditions = self.__read_data__(train_file, self.conditions_colnames)
            conditions = np.vstack({tuple(row) for row in conditions})
            #self.conditions = np.unique(self.conditions, axis=0)
        row_nm = np.random.choice(conditions.shape[0], num_sample_sim, replace=True)
        conditions = conditions[row_nm,:]
        '''
        with tf.device(self.device_name):
            self.generator_AtoB(is_training=True)
        self.saver = tf.train.Saver(max_to_keep=10000)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        #read the model and sample the data
        if not os.path.exists(self.model_folder):
            warning("Warning: there are no check points in model folder for data generation")
        else:
            ckpt_nm = os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(using_iter) + '.ckpt')
            self.saver.restore(self.sess, ckpt_nm)
        data_iterator = pd.read_csv(train_file, chunksize = 500000)
        data_batch = next(data_iterator)
        batchsize = 5000

        start_index = 0
        append = False
        sample_size = 0
        while sample_size < num_sample_sim:
            if start_index < len(data_batch):
                end_index = min(start_index + batchsize, len(data_batch))
                A_mb = data_batch.iloc[start_index:end_index][self.colnames_A].values
                start_index = end_index
            else:
                data_batch = next(data_iterator)
                start_index = 0
                end_index = min(start_index + batchsize, len(data_batch))
                A_mb = data_batch.iloc[start_index:end_index][self.colnames_A].values
                start_index = end_index
            gen_B = self.sess.run(
                self.gen_B,
                feed_dict={self.input_A: A_mb}
            )
            G_sample_batch_df = pd.DataFrame(gen_B, columns=self.colnames_B)
            sample_size += len(G_sample_batch_df)
            if not append:
                G_sample_batch_df.to_csv(output_file, index=False)
                append = True
            else:
                G_sample_batch_df.to_csv(output_file, mode='a', index=False, header=False)
            gc.collect()

class CDRAGAN2(GAN):
    def __init__(self, header_file, json_file, model_folder, log_file, device):
        GAN.__init__(self, header_file, json_file, model_folder, log_file, device)
        # inititalize logger file
        self.__init_logger__(log_file, 'cdragan')
        header = pd.read_csv(header_file)
        ix = [i for i,x in enumerate(header.iloc[0]) if x in ['N']]
        self.colnames = header.columns[ix]
        self.X_dim = len(self.colnames)
        ix = [i for i,x in enumerate(header.iloc[0]) if x in ['G']]
        self.conditions_colnames = header.columns[ix]
        self.y_dim = len(self.conditions_colnames)

    def costs(self):
        self.D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(self.D_real)))
        self.D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
        self.disc_cost = self.D_loss_real + self.D_loss_fake

        # define loss function for generator here because it is based on discriminator
        self.gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))

        # Gradient penalty
        self.alpha = tf.random_uniform(
            shape=[self.mb_size, 1],
            minval=0.,
            maxval=1.
        )
        self.differences = self.X_p - self.X
        self.X_interpolates = self.X + (self.alpha * self.differences)
        self.differences = self.y_p - self.y
        self.y_interpolates = self.y + (self.alpha * self.differences)
        self.interpolates = tf.concat(values=[self.X_interpolates, self.y_interpolates], axis=1)
        self.gradients = tf.gradients(self.__forward_step__(self.D_layers, self.interpolates, scope='D'), [self.interpolates])[0]
        self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), reduction_indices=[1]))
        self.gradient_penalty = tf.reduce_mean((self.slopes - 1.) ** 2)
        self.disc_cost += self.lambd * self.gradient_penalty

        # define optimizers
        self.gen_train_op = self.__init_optimizer__(self.G_optimizer, self.G_optimizer_param, self.gen_cost, scope='G')
        self.disc_train_op = self.__init_optimizer__(self.D_optimizer, self.D_optimizer_param, self.disc_cost, scope='D')

    def generator(self, is_training=True):
        # z is a placeholder, which is the sampled data....
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])
        self.__init_weights__(self.G_layers, self.z_dim + self.y_dim, output_dim=self.X_dim, scope='G')
        self.G_sample = self.__forward_step__(self.G_layers, tf.concat(values=[self.z, self.y], axis=1), scope='G')

    def discriminator(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim])
        self.X_p = tf.placeholder(tf.float32, shape=[None, self.X_dim])
        self.y_p = tf.placeholder(tf.float32, shape=[None, self.y_dim])
        self.__init_weights__(self.D_layers, self.X_dim + self.y_dim, scope='D')

        self.D_real = self.__forward_step__(self.D_layers, tf.concat(values=[self.X, self.y], axis=1), scope='D')
        self.D_fake = self.__forward_step__(self.D_layers, tf.concat(values=[self.G_sample, self.y], axis=1), scope='D')

    def fit(self, train_file):
        print("CDRAGAN training started")
        print("Number of iterations to run: {0:,}".format(self.num_iter))
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
            self.warm_start_iter = 0
        train_file_ext = os.path.basename(train_file).split('.')[-1]
        if train_file_ext == 'csv':
            data_source = ChunkedDataFrame(train_file, chunk_size = 500*self.mb_size, batch_size = self.mb_size, coverage = 10, reuse_file = True)
        elif train_file_ext == 'h5':
            data_source = HDFDataFrame(train_file, self.mb_size)
        with tf.device(self.device_name):
            self.generator(is_training=True)
            self.discriminator()
            self.costs()
        self.saver = tf.train.Saver(max_to_keep=10000)
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init)
        if self.warm_start_iter != 0:
            ckpt_nm_start = os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(self.warm_start_iter) + '.ckpt')
            self.saver.restore(self.sess, ckpt_nm_start)
            print("Warm start is used, warm_start_iter = {0:,}".format(self.warm_start_iter))

        plotD = []
        plotG = []
        plotGM = []
        plotDM = []
        it_vec = []
        start_index = 0
        end_index = 0
        for it in range(1 + self.warm_start_iter, self.num_iter + self.warm_start_iter + 1):
            end_index += 1
            _, y_mb = data_source.get_data_batch([self.colnames, self.conditions_colnames])
            if y_mb is None:
                print("Reached the end of data file, will stop training")
                return

            for _ in range(self.num_gen_per_disc):
                # training of generator....
                _, G_loss_curr = self.sess.run(
                    [self.gen_train_op, self.gen_cost],
                    feed_dict={self.y: y_mb, self.z: self.sample_z(self.mb_size, self.z_dim)}
                )

            X_mb, y_mb = data_source.get_data_batch([self.colnames, self.conditions_colnames])
            if X_mb is None:
                print("Reached the end of data file, will stop training")
                return
            X_mb_p = self.get_perturbed_batch(X_mb)
            y_mb_p = self.get_perturbed_batch(y_mb)
            for _ in range(self.num_disc_per_gen):
                _, D_loss_curr, penalty = self.sess.run(
                    [self.disc_train_op, self.disc_cost, self.gradient_penalty],
                    feed_dict={self.X: X_mb,
                               self.X_p: X_mb_p,
                               self.y: y_mb,
                               self.y_p: y_mb_p,
                               self.z: self.sample_z(self.mb_size, self.z_dim)}
                )

            # record the loss for every iteration!
            plotD.append(D_loss_curr)
            plotG.append(G_loss_curr)
            #plotDM.append(D_metric_curr)
            #plotGM.append(G_metric_curr)
            it_vec.append(it)

            if it % self.save_model_iter == 0:
                # save the model!
                self.saver.save(self.sess, os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(it) + '.ckpt'))

            if it % self.update_check_size == 0:
                # every certain learning steps, have an update check......
                if self.plot_ind == 1:
                    self.plot_errors(plotG, plotD, it_vec)
                self.logger.info('Iter: {}; D loss: {:.4}; G loss: {:.4}; penalty: {:.4}'.format(it, np.mean(plotD[start_index: end_index]), np.mean(plotG[start_index: end_index]), penalty))
                start_index = end_index

    def generate(self, train_file, output_file, num_sample_sim, using_iter, how='as_train'):
        '''
        if how == 'as_train':
            conditions = self.__read_data__(train_file, self.conditions_colnames)
        if how == 'uniform':
            conditions = self.__read_data__(train_file, self.conditions_colnames)
            conditions = np.vstack({tuple(row) for row in conditions})
            #self.conditions = np.unique(self.conditions, axis=0)
        row_nm = np.random.choice(conditions.shape[0], num_sample_sim, replace=True)
        conditions = conditions[row_nm,:]
        '''
        with tf.device(self.device_name):
            self.generator(is_training=False)
        self.saver = tf.train.Saver(max_to_keep=10000)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        #read the model and sample the data
        if not os.path.exists(self.model_folder):
            warning("Warning: there are no check points in model folder for data generation")
        else:
            ckpt_nm = os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(using_iter) + '.ckpt')
            self.saver.restore(self.sess, ckpt_nm)
        batchsize = 5000
        chunksize = 500000
        data_iterator = pd.read_csv(train_file, chunksize = chunksize, header=0)
        data_chunk = next(data_iterator)
        if how != 'as_train':
            data_chunk[how] = data_chunk[how].max()

        start_index = 0
        append = False
        sample_size = 0
        while sample_size < num_sample_sim:
            if start_index < len(data_chunk):
                end_index = min(start_index + min(batchsize, num_sample_sim - sample_size), len(data_chunk))
                y_sample = data_chunk.iloc[start_index:end_index][self.conditions_colnames].values
                start_index = end_index
            else:
                try:
                    data_chunk = next(data_iterator)
                except StopIteration:
                    data_iterator = pd.read_csv(train_file, chunksize = chunksize, header = 0)
                    data_chunk = next(data_iterator)
                if how != 'as_train':
                    data_chunk[how] = data_chunk[how].max()
                start_index = 0
                end_index = min(start_index + min(batchsize, num_sample_sim - sample_size), len(data_chunk))
                y_sample = data_chunk.iloc[start_index:end_index][self.conditions_colnames].values
                start_index = end_index
            z_sample = self.sample_z(y_sample.shape[0], self.z_dim)
            G_sample_batch = self.sess.run(
                self.G_sample,
                feed_dict={self.y: y_sample, self.z: z_sample}
            )
            G_sample_batch_df = pd.DataFrame(G_sample_batch, columns=self.colnames)
            G_sample_batch_df = pd.concat([G_sample_batch_df, pd.DataFrame(y_sample, columns=self.conditions_colnames)], axis=1, ignore_index=True)
            G_sample_batch_df.columns = [x for x in self.colnames] + [x for x in self.conditions_colnames]
            sample_size += len(G_sample_batch_df)
            if not append:
                G_sample_batch_df.to_csv(output_file, index=False)
                append = True
            else:
                G_sample_batch_df.to_csv(output_file, mode='a', index=False, header=False)
            gc.collect()

    def generate_feat(self, train_file, output_file, num_sample_sim, using_iter, how='as_train'):
        with tf.device(self.device_name):
            self.generator(is_training=False)
        self.saver = tf.train.Saver(max_to_keep=10000)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        #read the model and sample the data
        if not os.path.exists(self.model_folder):
            warning("Warning: there are no check points in model folder for data generation")
        else:
            ckpt_nm = os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(using_iter) + '.ckpt')
            self.saver.restore(self.sess, ckpt_nm)
        data_iterator = pd.read_csv(train_file, chunksize = 500000)
        data_batch = next(data_iterator)
        if how != 'as_train':
            data_batch[how] = data_batch[how].min()
        batchsize = 5000

        start_index = 0
        append = False
        sample_size = 0
        while sample_size < num_sample_sim:
            if start_index < len(data_batch):
                end_index = min(start_index + batchsize, len(data_batch))
                y_sample = data_batch.iloc[start_index:end_index][self.conditions_colnames].values
                X_sample = data_batch.iloc[start_index:end_index][self.colnames].values
                output = data_batch.iloc[start_index:end_index][['pkey','lift_ind']]
                start_index = end_index
            else:
                data_batch = next(data_iterator)
                if how != 'as_train':
                    data_batch[how] = data_batch[how].max()
                start_index = 0
                end_index = min(start_index + batchsize, len(data_batch))
                y_sample = data_batch.iloc[start_index:end_index][self.conditions_colnames].values
                X_sample = data_batch.iloc[start_index:end_index][self.colnames].values
                output = data_batch.iloc[start_index:end_index][['pkey','lift_ind']]
                start_index = end_index
            z_sample = self.sample_z(y_sample.shape[0], self.z_dim)
            G_sample_batch = self.sess.run(
                self.G_sample,
                feed_dict={self.y: y_sample, self.z: z_sample}
            )
            output['error'] = np.sum(np.abs(X_sample - G_sample_batch), axis=1)
            sample_size += len(output)
            if not append:
                output.to_csv(output_file, index=False)
                append = True
            else:
                output.to_csv(output_file, mode='a', index=False, header=False)
            gc.collect()

class CDRAGAN(GAN):
    def __init__(self, header_file, json_file, model_folder, log_file, device):
        GAN.__init__(self, header_file, json_file, model_folder, log_file, device)
        # inititalize logger file
        self.__init_logger__(log_file, 'cdragan')
        header = pd.read_csv(header_file)
        ix = [i for i,x in enumerate(header.iloc[0]) if x in ['N']]
        self.colnames = header.columns[ix]
        self.X_dim = len(self.colnames)
        ix = [i for i,x in enumerate(header.iloc[0]) if x in ['G']]
        self.conditions_colnames = header.columns[ix]
        self.y_dim = len(self.conditions_colnames)

    def costs(self):
        self.D_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real, labels=tf.ones_like(self.D_real)))
        self.D_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.zeros_like(self.D_fake)))
        self.disc_cost = self.D_loss_real + self.D_loss_fake

        # define loss function for generator here because it is based on discriminator
        self.gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake, labels=tf.ones_like(self.D_fake)))

        # Gradient penalty
        self.alpha = tf.random_uniform(
            shape=[self.mb_size, 1],
            minval=0.,
            maxval=1.
        )
        self.differences = self.X_p - self.X
        self.X_interpolates = self.X + (self.alpha * self.differences)
        self.differences = self.y_p - self.y
        self.y_interpolates = self.y + (self.alpha * self.differences)
        self.interpolates = tf.concat(values=[self.X_interpolates, self.y_interpolates], axis=1)
        self.gradients = tf.gradients(self.__forward_step__(self.D_layers, self.interpolates, scope='D'), [self.interpolates])[0]
        self.slopes = tf.sqrt(tf.reduce_sum(tf.square(self.gradients), reduction_indices=[1]))
        self.gradient_penalty = tf.reduce_mean((self.slopes - 1.) ** 2)
        self.disc_cost += self.lambd * self.gradient_penalty

        # define optimizers
        self.gen_train_op = self.__init_optimizer__(self.G_optimizer, self.G_optimizer_param, self.gen_cost, scope='G')
        self.disc_train_op = self.__init_optimizer__(self.D_optimizer, self.D_optimizer_param, self.disc_cost, scope='D')

    def generator(self, is_training=True):
        # z is a placeholder, which is the sampled data....
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.y_dim])
        self.__init_weights__(self.G_layers, self.z_dim + self.y_dim, output_dim=self.X_dim, scope='G')
        self.G_sample = self.__forward_step__(self.G_layers, tf.concat(values=[self.z, self.y], axis=1), scope='G')

    def discriminator(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim])
        self.X_p = tf.placeholder(tf.float32, shape=[None, self.X_dim])
        self.y_p = tf.placeholder(tf.float32, shape=[None, self.y_dim])
        self.__init_weights__(self.D_layers, self.X_dim + self.y_dim, scope='D')

        self.D_real = self.__forward_step__(self.D_layers, tf.concat(values=[self.X, self.y], axis=1), scope='D')
        self.D_fake = self.__forward_step__(self.D_layers, tf.concat(values=[self.G_sample, self.y], axis=1), scope='D')

    def fit(self, train_file):
        print("CDRAGAN training started")
        print("Number of iterations to run: {0:,}".format(self.num_iter))
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
            self.warm_start_iter = 0
        train_file_ext = os.path.basename(train_file).split('.')[-1]
        if train_file_ext == 'csv':
            data_source = ChunkedDataFrame(train_file, chunk_size = 500*self.mb_size, batch_size = self.mb_size, coverage = 10, reuse_file = True)
        elif train_file_ext == 'h5':
            data_source = HDFDataFrame(train_file, self.mb_size)
        with tf.device(self.device_name):
            self.generator(is_training=True)
            self.discriminator()
            self.costs()
        self.saver = tf.train.Saver(max_to_keep=10000)
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init)
        if self.warm_start_iter != 0:
            ckpt_nm_start = os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(self.warm_start_iter) + '.ckpt')
            self.saver.restore(self.sess, ckpt_nm_start)
            print("Warm start is used, warm_start_iter = {0:,}".format(self.warm_start_iter))

        plotD = []
        plotG = []
        plotGM = []
        plotDM = []
        it_vec = []
        start_index = 0
        end_index = 0
        for it in range(1 + self.warm_start_iter, self.num_iter + self.warm_start_iter + 1):
            end_index += 1
            _, y_mb = data_source.get_data_batch([self.colnames, self.conditions_colnames])
            if y_mb is None:
                print("Reached the end of data file, will stop training")
                return

            for _ in range(self.num_gen_per_disc):
                # training of generator....
                _, G_loss_curr = self.sess.run(
                    [self.gen_train_op, self.gen_cost],
                    feed_dict={self.y: y_mb, self.z: self.sample_z(self.mb_size, self.z_dim)}
                )

            X_mb, y_mb = data_source.get_data_batch([self.colnames, self.conditions_colnames])
            if X_mb is None:
                print("Reached the end of data file, will stop training")
                return
            X_mb_p = self.get_perturbed_batch(X_mb)
            y_mb_p = self.get_perturbed_batch(y_mb)
            for _ in range(self.num_disc_per_gen):
                _, D_loss_curr, penalty = self.sess.run(
                    [self.disc_train_op, self.disc_cost, self.gradient_penalty],
                    feed_dict={self.X: X_mb,
                               self.X_p: X_mb_p,
                               self.y: y_mb,
                               self.y_p: y_mb_p,
                               self.z: self.sample_z(self.mb_size, self.z_dim)}
                )

            # record the loss for every iteration!
            plotD.append(D_loss_curr)
            plotG.append(G_loss_curr)
            #plotDM.append(D_metric_curr)
            #plotGM.append(G_metric_curr)
            it_vec.append(it)

            if it % self.save_model_iter == 0:
                # save the model!
                self.saver.save(self.sess, os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(it) + '.ckpt'))

            if it % self.update_check_size == 0:
                # every certain learning steps, have an update check......
                if self.plot_ind == 1:
                    self.plot_errors(plotG, plotD, it_vec)
                self.logger.info('Iter: {}; D loss: {:.4}; G loss: {:.4}; penalty: {:.4}'.format(it, np.mean(plotD[start_index: end_index]), np.mean(plotG[start_index: end_index]), penalty))
                start_index = end_index

    def generate(self, train_file, output_file, num_sample_sim, using_iter, how='as_train'):
        '''
        if how == 'as_train':
            conditions = self.__read_data__(train_file, self.conditions_colnames)
        if how == 'uniform':
            conditions = self.__read_data__(train_file, self.conditions_colnames)
            conditions = np.vstack({tuple(row) for row in conditions})
            #self.conditions = np.unique(self.conditions, axis=0)
        row_nm = np.random.choice(conditions.shape[0], num_sample_sim, replace=True)
        conditions = conditions[row_nm,:]
        '''
        with tf.device(self.device_name):
            self.generator(is_training=False)
        self.saver = tf.train.Saver(max_to_keep=10000)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        #read the model and sample the data
        if not os.path.exists(self.model_folder):
            warning("Warning: there are no check points in model folder for data generation")
        else:
            ckpt_nm = os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(using_iter) + '.ckpt')
            self.saver.restore(self.sess, ckpt_nm)
        batchsize = 5000
        chunksize = 500000
        data_iterator = pd.read_csv(train_file, chunksize = chunksize, header=0)
        data_chunk = next(data_iterator)
        if how != 'as_train':
            data_chunk[how] = data_chunk[how].max()

        start_index = 0
        append = False
        sample_size = 0
        while sample_size < num_sample_sim:
            if start_index < len(data_chunk):
                end_index = min(start_index + min(batchsize, num_sample_sim - sample_size), len(data_chunk))
                y_sample = data_chunk.iloc[start_index:end_index][self.conditions_colnames].values
                start_index = end_index
            else:
                try:
                    data_chunk = next(data_iterator)
                except StopIteration:
                    data_iterator = pd.read_csv(train_file, chunksize = chunksize, header = 0)
                    data_chunk = next(data_iterator)
                if how != 'as_train':
                    data_chunk[how] = data_chunk[how].max()
                start_index = 0
                end_index = min(start_index + min(batchsize, num_sample_sim - sample_size), len(data_chunk))
                y_sample = data_chunk.iloc[start_index:end_index][self.conditions_colnames].values
                start_index = end_index
            z_sample = self.sample_z(y_sample.shape[0], self.z_dim)
            G_sample_batch = self.sess.run(
                self.G_sample,
                feed_dict={self.y: y_sample, self.z: z_sample}
            )
            G_sample_batch_df = pd.DataFrame(G_sample_batch, columns=self.colnames)
            G_sample_batch_df = pd.concat([G_sample_batch_df, pd.DataFrame(y_sample, columns=self.conditions_colnames)], axis=1, ignore_index=True)
            G_sample_batch_df.columns = [x for x in self.colnames] + [x for x in self.conditions_colnames]
            sample_size += len(G_sample_batch_df)
            if not append:
                G_sample_batch_df.to_csv(output_file, index=False)
                append = True
            else:
                G_sample_batch_df.to_csv(output_file, mode='a', index=False, header=False)
            gc.collect()

    def generate_feat(self, train_file, output_file, num_sample_sim, using_iter, how='as_train'):
        with tf.device(self.device_name):
            self.generator(is_training=False)
        self.saver = tf.train.Saver(max_to_keep=10000)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        #read the model and sample the data
        if not os.path.exists(self.model_folder):
            warning("Warning: there are no check points in model folder for data generation")
        else:
            ckpt_nm = os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(using_iter) + '.ckpt')
            self.saver.restore(self.sess, ckpt_nm)
        data_iterator = pd.read_csv(train_file, chunksize = 500000)
        data_batch = next(data_iterator)
        if how != 'as_train':
            data_batch[how] = data_batch[how].min()
        batchsize = 5000

        start_index = 0
        append = False
        sample_size = 0
        while sample_size < num_sample_sim:
            if start_index < len(data_batch):
                end_index = min(start_index + batchsize, len(data_batch))
                y_sample = data_batch.iloc[start_index:end_index][self.conditions_colnames].values
                X_sample = data_batch.iloc[start_index:end_index][self.colnames].values
                output = data_batch.iloc[start_index:end_index][['pkey','lift_ind']]
                start_index = end_index
            else:
                data_batch = next(data_iterator)
                if how != 'as_train':
                    data_batch[how] = data_batch[how].max()
                start_index = 0
                end_index = min(start_index + batchsize, len(data_batch))
                y_sample = data_batch.iloc[start_index:end_index][self.conditions_colnames].values
                X_sample = data_batch.iloc[start_index:end_index][self.colnames].values
                output = data_batch.iloc[start_index:end_index][['pkey','lift_ind']]
                start_index = end_index
            z_sample = self.sample_z(y_sample.shape[0], self.z_dim)
            G_sample_batch = self.sess.run(
                self.G_sample,
                feed_dict={self.y: y_sample, self.z: z_sample}
            )
            output['error'] = np.sum(np.abs(X_sample - G_sample_batch), axis=1)
            sample_size += len(output)
            if not append:
                output.to_csv(output_file, index=False)
                append = True
            else:
                output.to_csv(output_file, mode='a', index=False, header=False)
            gc.collect()

class WGAN(GAN):
    def __init__(self, header_file, json_file, model_folder, log_file, device):
        GAN.__init__(self, header_file, json_file, model_folder, log_file, device)
        # inititalize logger file
        self.__init_logger__(log_file, 'wgan')
        header = pd.read_csv(header_file)
        ix = [i for i,x in enumerate(header.iloc[0]) if x in ['N','G']]
        self.colnames = header.columns[ix]
        self.X_dim = len(self.colnames)

    def costs(self):
        # define cost functions
        self.disc_cost = tf.reduce_mean(self.D_fake) - tf.reduce_mean(self.D_real)
        self.gen_cost = -tf.reduce_mean(self.D_fake)

        # define optimizers
        self.gen_train_op = self.__init_optimizer__(self.G_optimizer, self.G_optimizer_param, self.gen_cost, scope='G')
        self.disc_train_op = self.__init_optimizer__(self.D_optimizer, self.D_optimizer_param, self.disc_cost, scope='D')

        # clip weights of discriminator
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'D')
        self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in var_list]

    def generator(self, is_training=True):
        # z is a placeholder, which is the sampled data....
        self.z = tf.placeholder(tf.float32, shape=[None, self.z_dim])
        self.__init_weights__(self.G_layers, self.z_dim, output_dim=self.X_dim, scope='G')
        self.G_sample = self.__forward_step__(self.G_layers, self.z, scope='G')

    def discriminator(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.X_dim])
        self.X_p = tf.placeholder(tf.float32, shape=[None, self.X_dim])
        self.__init_weights__(self.D_layers, self.X_dim, scope='D')

        self.D_real = self.__forward_step__(self.D_layers, self.X, scope='D')
        self.D_fake = self.__forward_step__(self.D_layers, self.G_sample, scope='D')

    def fit(self, train_file):
        print("WGAN training started")
        print("Number of iterations to run: {0:,}".format(self.num_iter))
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
            self.warm_start_iter = 0
        data = self.__read_data__(train_file, self.colnames)
        data_size = data.shape[0]
        with tf.device(self.device_name):
            self.generator(is_training=True)
            self.discriminator()
            self.costs()
        self.saver = tf.train.Saver(max_to_keep=10000)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if self.warm_start_iter != 0:
            ckpt_nm_start = os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(self.warm_start_iter) + '.ckpt')
            self.saver.restore(self.sess, ckpt_nm_start)
            print("Warm start is used, warm_start_iter = {0:,}".format(self.warm_start_iter))

        plotD = []
        plotG = []
        it_vec = []
        start_index = 0
        end_index = 0
        for it in range(1 + self.warm_start_iter, self.num_iter + self.warm_start_iter + 1):
            end_index += 1
            for _ in range(self.num_gen_per_disc):
                # training of generator....
                _, G_loss_curr = self.sess.run(
                    [self.gen_train_op, self.gen_cost],
                    feed_dict={self.z: self.sample_z(self.mb_size, self.z_dim)}
                )

            X_mb = data[self.next_batch(data_size, self.mb_size),:]
            X_mb_p = self.get_perturbed_batch(X_mb)

            for _ in range(self.num_disc_per_gen):
                _, D_loss_curr, _ = self.sess.run(
                    [self.disc_train_op, self.disc_cost, self.clip_D],
                    feed_dict={self.X: X_mb, self.X_p: X_mb_p, self.z: self.sample_z(self.mb_size, self.z_dim)}
                )

            # record the loss for every iteration!
            plotD.append(D_loss_curr)
            plotG.append(G_loss_curr)
            it_vec.append(it)
            if it % self.save_model_iter == 0:
                # save the model!
                self.saver.save(self.sess, os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(it) + '.ckpt'))
            if it % self.update_check_size == 0:
                # every certain learning steps, have an update check......
                if self.plot_ind == 1:
                    self.plot_errors(plotG, plotD, it_vec)
                self.logger.info('Iter: {}; D loss: {:.4}; G loss: {:.4}'.format(it, np.mean(plotD[start_index: end_index]), np.mean(plotG[start_index: end_index])))
                start_index = end_index

    def generate(self, output_file, num_sample_sim, using_iter):
        with tf.device(self.device_name):
            self.generator(is_training=False)
        self.saver = tf.train.Saver(max_to_keep=10000)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        #read the model and sample the data!!!!
        if not os.path.exists(self.model_folder):
            warning("Warning: there are no check points in model folder for data generation")
        else:
            ckpt_nm = os.path.join(self.model_folder, self.checkpoint_nm + '_' + str(using_iter) + '.ckpt')
            self.saver.restore(self.sess, ckpt_nm)
        G_sample = self.sess.run(
            self.G_sample,
            feed_dict={self.z: self.sample_z(num_sample_sim, self.z_dim)}
        )
        G_sample_df = pd.DataFrame(G_sample, columns=self.colnames)
        G_sample_df.reset_index(drop=True, inplace=True)
        G_sample_df.to_csv(output_file, index=False)

