'''
Deep Highway Network

Adopted from @awjuliani by @MWransky

'''
import tensorflow as tf
import tensorflow.contrib.slim as slim

num_labels = 99
reg_parameter = 0.001
learn_rate = 0.001
# total layers need to be divisible by 5
total_layers = 15
units_between_stride = int(total_layers / 5)


class Network():
    def __init__(self):
        # The network recieves a batch of images
        self.input_layer = tf.placeholder(shape=[None, 80, 80, 1], dtype=tf.float32, name='input')
        self.label_layer = tf.placeholder(shape=[None], dtype=tf.int32)
        self.label_oh = slim.layers.one_hot_encoding(self.label_layer, num_labels)
        # initial layer fed with batch images
        self.layer = slim.conv2d(self.input_layer, 64, [3, 3],
            normalizer_fn=slim.batch_norm, weights_regularizer=slim.l2_regularizer(reg_parameter),
            biases_regularizer=slim.l2_regularizer(reg_parameter), scope='conv_'+str(0))
        # build out the highway net units
        for i in range(5):
            for j in range(units_between_stride):
                self.layer = highwayUnit(self.layer, j+(i*units_between_stride))
            self.layer = slim.conv2d(self.layer, 64, [3, 3],
                normalizer_fn=slim.batch_norm, weights_regularizer=slim.l2_regularizer(reg_parameter),
                biases_regularizer=slim.l2_regularizer(reg_parameter), scope='conv_s_'+str(i))
        # extract transition layer
        # self.top = slim.conv2d(self.layer, num_labels, [3, 3],
            # normalizer_fn=slim.batch_norm, activation_fn=None, scope='conv_top')
        self.top = slim.fully_connected(slim.layers.flatten(self.layer), num_labels,
            normalizer_fn=slim.batch_norm,
            weights_regularizer=slim.l2_regularizer(reg_parameter),
            biases_regularizer=slim.l2_regularizer(reg_parameter),
            activation_fn=None, scope='fully_connected_top')
        # generate softmax probabilities
        self.probs = slim.layers.softmax(self.top)
        # calculate reduce mean loss function
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(self.top, self.label_oh))
        # self.loss = tf.reduce_mean(-tf.reduce_sum(self.label_oh * tf.log(self.output) + 1e-10, reduction_indices=[1]))
        # optimizer
        self.trainer = tf.train.AdamOptimizer(learning_rate=learn_rate)
        # minimization
        self.update = self.trainer.minimize(self.loss)


def highwayUnit(input_layer, i):
    with tf.variable_scope("highway_unit"+str(i)):
        H = slim.conv2d(input_layer, 64, [3, 3])
        # Push the network to use the skip connection via a negative init
        T = slim.conv2d(input_layer, 64, [3, 3],
            biases_initializer=tf.constant_initializer(-1.0),
            weights_regularizer=slim.l2_regularizer(reg_parameter),
            biases_regularizer=slim.l2_regularizer(reg_parameter),
            activation_fn=tf.nn.sigmoid)
        output = H*T + input_layer*(1.0-T)
        return output
