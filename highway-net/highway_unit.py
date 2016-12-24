'''
2D Convolutional Highway Unit

Inspired by @trangptm

Modified and adapted by @MWransky

'''

import numpy as np

from keras.layers import *
from keras import backend as K
from keras.engine.topology import Layer


class HighwayUnit(Layer):
    def __init__(self, nb_filter=32, nb_row=3, nb_col=3, transform_bias=-1,
                 init='glorot_uniform', activation='relu', weights=None,
                 border_mode='same', subsample=(1, 1), dim_ordering='tf',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 bias=True, **kwargs):

        # check that convolution border is accepted type
        if border_mode not in {'valid', 'same'}:
            raise Exception('Invalid border mode for Convolution2D:', border_mode)
        self.nb_filter = nb_filter
        self.nb_row = nb_row
        self.nb_col = nb_col
        self.transform_bias = transform_bias
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.activation = activations.get(activation)
        self.border_mode = border_mode
        self.subsample = tuple(subsample)
        self.dim_ordering = dim_ordering

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.bias = bias
        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        super(HighwayUnit, self).__init__(**kwargs)

    def build(self, input_shape):
        stack_size = input_shape[3]
        self.W_shape = (self.nb_row, self.nb_col, stack_size, self.nb_filter)

        self.W = self.init(self.W_shape, name='{}_W'.format(self.name))
        self.W_gate = self.init(self.W_shape, name='{}_W_carry'.format(self.name))

        # set up trainable weights
        if self.bias:
            self.b = K.zeros((self.nb_filter,), name='{}_b'.format(self.name))
            self.b_gate = K.variable(np.ones(self.nb_filter,), name='{}_b_gate'.format(self.name))
            self.trainable_weights = [self.W, self.b, self.W_gate, self.b_gate]
        else:
            self.trainable_weights = [self.W, self.W_gate]
        self.regularizers = []

        # set up weight/bias regularizer parameters
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output_shape_for(self, input_shape):
        rows = input_shape[1]
        cols = input_shape[2]

        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])

        return (input_shape[0], rows, cols, self.nb_filter)

    def call(self, x, mask=None):
        # compute the candidate hidden state
        transform = K.conv2d(x, self.W, strides=self.subsample,
                          border_mode=self.border_mode,
                          dim_ordering=self.dim_ordering,
                          filter_shape=self.W_shape)
        if self.bias:
            transform += K.reshape(self.b, (1, 1, 1, self.nb_filter))
        transform = self.activation(transform)

        transform_gate = K.conv2d(x, self.W_gate, strides=self.subsample,
                          border_mode=self.border_mode,
                          dim_ordering=self.dim_ordering,
                          filter_shape=self.W_shape)
        if self.bias:
            transform_gate += K.reshape(self.b_gate, (1, 1, 1, self.nb_filter))
        transform_gate = K.sigmoid(transform_gate)
        carry_gate = 1.0 - transform_gate

        return transform * transform_gate + x * carry_gate

    # Define get_config method so load_from_json can run
    def get_config(self):
        config = {'nb_filter': self.nb_filter,
                  'nb_row': self.nb_row,
                  'nb_col': self.nb_col,
                  'transform_bias': self.transform_bias,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'subsample': self.subsample,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'bias': self.bias}
        base_config = super(HighwayUnit, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
