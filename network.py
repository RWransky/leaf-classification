import tensorflow as tf
import tensorflow.contrib.slim as slim


class Network():
    def __init__(self, h_size, rnn_cell, network_name):
        # The network recieves a frame from the game, flattened into an array
        # It then resizes it and processes it through four convolutional layers
        self.scalarInput = tf.placeholder(shape=[None, 80*80], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 80, 80, 1])
        self.conv1 = slim.convolution2d(inputs=self.imageIn, num_outputs=32,
            kernel_size=[8, 8], stride=[4, 4], padding='VALID',
            biases_initializer=None, scope=network_name+'_conv1')
        self.conv2 = slim.convolution2d(
            inputs=self.conv1, num_outputs=64,
            kernel_size=[4, 4], stride=[2, 2], padding='VALID',
            biases_initializer=None, scope=network_name+'_conv2')
        self.conv3 = slim.convolution2d(
            inputs=self.conv2, num_outputs=64,
            kernel_size=[3, 3], stride=[1, 1], padding='VALID',
            biases_initializer=None, scope=network_name+'_conv3')
        self.conv4 = slim.convolution2d(
            inputs=self.conv3, num_outputs=512,
            kernel_size=[6, 6], stride=[1, 1], padding='VALID',
            biases_initializer=None, scope=network_name+'_conv4')

        self.trainLength = tf.placeholder(dtype=tf.int32)
        # We take the output from the final convolutional layer and send it to a recurrent layer.
        # The input must be reshaped into [batch x trace x units] for rnn processing,
        # and then returned to [batch x units] when sent through the upper levles.
        self.batch_size = tf.placeholder(dtype=tf.int32)
        self.convFlat = tf.reshape(slim.flatten(self.conv4), [self.batch_size, self.trainLength, h_size])
        self.state_in = rnn_cell.zero_state(self.batch_size, tf.float32)
        self.rnn, self.rnn_state = tf.nn.dynamic_rnn(
                inputs=self.convFlat, cell=rnn_cell, dtype=tf.float32,
                initial_state=self.state_in, scope=network_name+'_rnn')
        self.rnn = tf.reshape(self.rnn, shape=[-1, h_size])
        # The output from the recurrent player is then split into
        # separate Value and Advantage streams
        self.streamA, self.streamV = tf.split(1, 2, self.rnn)
        self.AW = tf.Variable(tf.random_normal([int(h_size/2), 99]))
        self.VW = tf.Variable(tf.random_normal([int(h_size/2), 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        self.salience = tf.gradients(self.Advantage, self.imageIn)
        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.sub(self.Advantage,
            tf.reduce_mean(self.Advantage, reduction_indices=1, keep_dims=True))
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 99, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.mul(self.Qout, self.actions_onehot), reduction_indices=1)

        self.td_error = tf.square(self.targetQ - self.Q)

        # In order to only propogate accurate gradients through the network, we will mask the first
        # half of the losses for each trace as per Lample & Chatlot 2016
        self.maskA = tf.zeros([self.batch_size, tf.to_int32(self.trainLength/2)])
        self.maskB = tf.ones([self.batch_size, tf.to_int32(self.trainLength/2)])
        self.mask = tf.concat(1, [self.maskA, self.maskB])
        self.mask = tf.reshape(self.mask, [-1])
        self.loss = tf.reduce_mean(self.td_error * self.mask)

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)
