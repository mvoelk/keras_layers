"""
SPDX-License-Identifier: MIT
Copyright © 2018 - 2024 Markus Völk
Code was taken from https://github.com/mvoelk/keras_layers
"""

import numpy as np
import tensorflow as tf
import warnings

from keras import backend as K
from keras.layers import Layer, Lambda
from keras.layers import InputSpec
from keras import initializers, regularizers, constraints, activations
#from keras.utils import conv_utils
from tensorflow.python.keras.utils import conv_utils


def normal_init(shape, dtype=None, partition_info=None):
    v = np.random.normal(0, 1, size=shape)
    v = np.clip(v, -3, +3)
    return K.constant(v, dtype=dtype)

def uniform_init(shape, dtype=None, partition_info=None):
    v = np.random.uniform(-3**0.5, 3**0.5, size=shape)
    return K.constant(v, dtype=dtype)

def orthogonal_init(shape, dtype=None, partition_info=None):
    m, n = np.prod(shape[:-1]), shape[-1]
    a = np.random.normal(size=(max(m,n), min(m,n)))
    q, r = np.linalg.qr(a)
    q = q / np.std(q)
    if m < n:
        q = np.transpose(q)
    v = np.reshape(q, shape)
    #v = np.clip(v, -3, +3)
    return K.constant(v, dtype=dtype)

def near_zero_init(shape, dtype=None, partition_info=None):
    eps = 1e-8
    v = np.random.uniform(-eps, eps, size=shape)
    return K.constant(v, dtype=dtype)

def conv_init_linear(shape, dtype=None, partition_info=None):
    v = np.random.normal(0, 1, size=shape)
    v = np.clip(v, -3, +3)
    fanin = np.prod(shape[:-1])
    v = v / (fanin**0.5)
    return K.constant(v, dtype=dtype)

def conv_init_relu(shape, dtype=None, partition_info=None):
    # He init
    v = np.random.normal(0, 1, size=shape)
    v = np.clip(v, -3, +3)
    fanin = np.prod(shape[:-1])
    v = v / (fanin**0.5) * 2**0.5
    return K.constant(v, dtype=dtype)

def conv_init_relu2(shape, dtype=None, partition_info=None):
    v = np.random.normal(0, 1, size=shape)
    v = np.clip(v, -3, +3)
    fanin = np.prod(shape[:-1])
    v = v / (fanin**0.5) * 2
    return K.constant(v, dtype=dtype)

def depthwiseconv_init_linear(shape, dtype=None, partition_info=None):
    v = np.random.normal(0, 1, size=shape)
    v = np.clip(v, -3, +3)
    fanin = np.prod(shape[:-2])
    v = v / (fanin**0.5)
    return K.constant(v, dtype=dtype)

def depthwiseconv_init_relu(shape, dtype=None, partition_info=None):
    # He init
    v = np.random.normal(0, 1, size=shape)
    v = np.clip(v, -3, +3)
    fanin = np.prod(shape[:-2])
    v = v / (fanin**0.5) * 2**0.5
    return K.constant(v, dtype=dtype)

def orthogonal_conv_init_relu(shape, dtype=None, partition_info=None):
    fanin = np.prod(shape[:-1])
    return orthogonal_init(shape, dtype) / (fanin**0.5) * 2**0.5


class Conv1DBaseLayer(Layer):
    """Basic Conv1D class from which other layers inherit.
    """
    def __init__(self,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 #data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 weightnorm=False, equalize=False, eps=1e-6,
                 **kwargs):

        super(Conv1DBaseLayer, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

        self.rank = rank = 1
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.weightnorm = weightnorm
        self.equalize = equalize
        self.eps = eps

    def build(self, input_shape):

        self.kernel = self.add_weight(name='kernel',
                                      shape=self.kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True,
                                      dtype=self.dtype)
        
        if self.weightnorm:
            assert not self.equalize
            self.wn_g = self.add_weight(name='wn_g',
                                        shape=(self.filters,),
                                        initializer=initializers.Ones(),
                                        trainable=True,
                                        dtype=self.dtype)
        
        if self.equalize:
            assert not self.weightnorm
            if self.kernel_initializer not in [normal_init, uniform_init]:
                warnings.warn('when equalization is used, normal_init or uniform_init are the recommended kernel initializers')

            if hasattr(self, 'depth_multiplier'):
                fanin = np.prod(self.kernel_shape[:-2])
            else:
                fanin = np.prod(self.kernel_shape[:-1])
            self.scale = np.sqrt(2) / np.sqrt(fanin)

        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True,
                                        dtype=self.dtype)
        else:
            self.bias = None

        super(Conv1DBaseLayer, self).build(input_shape)

    def get_config(self):
        config = super(Conv1DBaseLayer, self).get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'weightnorm': self.weightnorm,
            'equalize': self.equalize,
            'eps': self.eps,
        })
        return config


class Conv1D(Conv1DBaseLayer):
    """Conv1D Layer with Weight Normalization and Equalized Learning Rates.
    
    # Arguments
        They are the same as for the normal Conv1D layer.
        weightnorm: Boolean flag, whether Weight Normalization is used or not.
        equalize: Boolean flag, wehter Equalized Learning Rates are used.
        
    # References
        [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](http://arxiv.org/abs/1602.07868)
    """
    def __init__(self, filters, kernel_size, **kwargs):
        super(Conv1D, self).__init__(kernel_size, **kwargs)
        
        self.filters = filters
    
    def build(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape
        
        self.kernel_shape = (*self.kernel_size, feature_shape[-1], self.filters)
        
        super(Conv1D, self).build(input_shape)
        
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            features = inputs[0]
        else:
            features = inputs
        
        kernel = self.kernel

        if self.weightnorm:
            norm = tf.sqrt(tf.reduce_sum(tf.square(kernel), (0,1)) + self.eps)
            kernel = kernel / norm * self.wn_g

        if self.equalize:
            kernel = kernel * self.scale

        features = K.conv1d(features, kernel,
                            strides=self.strides,
                            padding=self.padding,
                            dilation_rate=self.dilation_rate)
        
        if self.use_bias:
            features = tf.add(features, self.bias)
        
        if self.activation is not None:
            features = self.activation(features)
        
        return features

    def get_config(self):
        config = super(Conv1D, self).get_config()
        config.update({
            'filters': self.filters,
        })
        return config


class Conv2DBaseLayer(Layer):
    """Basic Conv2D class from which other layers inherit.
    """
    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 #data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 weightnorm=False, equalize=False, eps=1e-6,
                 **kwargs):

        super(Conv2DBaseLayer, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)

        self.rank = rank = 2
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.weightnorm = weightnorm
        self.equalize = equalize
        self.eps = eps

    def build(self, input_shape):

        self.kernel = self.add_weight(name='kernel',
                                      shape=self.kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True,
                                      dtype=self.dtype)
        
        if self.weightnorm:
            assert not self.equalize
            self.wn_g = self.add_weight(name='wn_g',
                                        shape=(self.filters,),
                                        initializer=initializers.Ones(),
                                        trainable=True,
                                        dtype=self.dtype)
        
        if self.equalize:
            assert not self.weightnorm
            if self.kernel_initializer not in [normal_init, uniform_init]:
                warnings.warn('when equalization is used, normal_init or uniform_init are the recommended kernel initializers')

            if hasattr(self, 'depth_multiplier'):
                fanin = np.prod(self.kernel_shape[:-2])
            else:
                fanin = np.prod(self.kernel_shape[:-1])
            self.scale = np.sqrt(2) / np.sqrt(fanin)

        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True,
                                        dtype=self.dtype)
        else:
            self.bias = None

        super(Conv2DBaseLayer, self).build(input_shape)

    def get_config(self):
        config = super(Conv2DBaseLayer, self).get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'weightnorm': self.weightnorm,
            'equalize': self.equalize,
            'eps': self.eps,
        })
        return config


class Conv2D(Conv2DBaseLayer):
    """Conv2D Layer with Weight Normalization and Equalized Learning Rates.
    
    # Arguments
        They are the same as for the normal Conv2D layer.
        weightnorm: Boolean flag, whether Weight Normalization is used or not.
        equalize: Boolean flag, wehter Equalized Learning Rates are used.
        
    # References
        [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](http://arxiv.org/abs/1602.07868)
    """
    def __init__(self, filters, kernel_size, **kwargs):
        super(Conv2D, self).__init__(kernel_size, **kwargs)
        
        self.filters = filters
    
    def build(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape
        
        self.kernel_shape = (*self.kernel_size, feature_shape[-1], self.filters)
        
        super(Conv2D, self).build(input_shape)
        
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            features = inputs[0]
        else:
            features = inputs
        
        kernel = self.kernel

        if self.weightnorm:
            norm = tf.sqrt(tf.reduce_sum(tf.square(kernel), (0,1,2)) + self.eps)
            kernel = kernel / norm * self.wn_g

        if self.equalize:
            kernel = kernel * self.scale

        features = K.conv2d(features, kernel,
                            strides=self.strides,
                            padding=self.padding,
                            dilation_rate=self.dilation_rate)
        
        if self.use_bias:
            features = tf.add(features, self.bias)
        
        if self.activation is not None:
            features = self.activation(features)
        
        return features

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.update({
            'filters': self.filters,
        })
        return config


class DepthwiseConv2D(Conv2DBaseLayer):
    """2D depthwise convolution layer.
    
    # Arguments
        They are the same as for the normal Conv2D layer except the 'depth_multiplier' 
        argument is used instead of 'filters'.
        depth_multiplier: Integer
        weightnorm: Boolean flag, whether Weight Normalization is used or not.
        equalize: Boolean flag, wehter Equalized Learning Rates are used.

    # Notes
        A DepthwiseConv2D layer followed by an 1x1 Conv2D layer is equivalent
        to the SeparableConv2D layer provided by Keras.
    
    # References
        [Xception: Deep Learning with Depthwise Separable Convolutions](http://arxiv.org/abs/1610.02357)
    """
    def __init__(self, depth_multiplier, kernel_size, kernel_initializer=depthwiseconv_init_relu, **kwargs):
        
        self.depth_multiplier = depth_multiplier

        kwargs['kernel_initializer'] = kernel_initializer
        super(DepthwiseConv2D, self).__init__(kernel_size, **kwargs)
    
    def build(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape
        
        self.filters = feature_shape[-1] * self.depth_multiplier
        self.kernel_shape = (*self.kernel_size, feature_shape[-1], self.depth_multiplier)

        super(DepthwiseConv2D, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            features = inputs[0]
        else:
            features = inputs
        
        kernel = self.kernel

        if self.weightnorm:
            norm = tf.sqrt(tf.reduce_sum(tf.square(kernel), (0,1)) + self.eps)
            kernel = kernel / norm * tf.reshape(self.wn_g, (-1, self.depth_multiplier))

        if self.equalize:
            kernel = kernel * self.scale

        features = K.depthwise_conv2d(features, kernel,
                                      strides=self.strides,
                                      padding=self.padding,
                                      dilation_rate=self.dilation_rate)
        
        if self.use_bias:
            features = tf.add(features, self.bias)
        
        if self.activation is not None:
            features = self.activation(features)
        
        return features
    
    def compute_output_shape(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape
        
        space = feature_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        
        feature_shape = [feature_shape[0], *new_space, feature_shape[-1]*self.depth_multiplier]
        
        return feature_shape
    
    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.update({
            'depth_multiplier': self.depth_multiplier,
        })
        return config


class SparseConv2D(Conv2DBaseLayer):
    """2D Sparse Convolution layer for sparse input data.
    
    # Arguments
        They are the same as for the normal Conv2D layer.
        binary: Boolean flag, whether the sparsity is propagated as binary 
            mask or as float values.
        weightnorm: Boolean flag, whether Weight Normalization is used or not.
        equalize: Boolean flag, wehter Equalized Learning Rates are used.
    
    # Input shape
        features: 4D tensor with shape (batch_size, rows, cols, channels)
        mask: 4D tensor with shape (batch_size, rows, cols, 1)
    
    # Example
        x, m = SparseConv2D(32, 3, padding='same')(x)
        x = Activation('relu')(x)
        x, m = SparseConv2D(32, 3, padding='same')([x,m])
        x = Activation('relu')(x)
    
    # Notes
        Sparse Convolution propagates the sparsity of the input data
        through the network using a 2D mask.
    
    # References
        [Sparsity Invariant CNNs](https://arxiv.org/abs/1708.06500)
    """
    def __init__(self, filters, kernel_size, kernel_initializer=conv_init_relu, binary=True, **kwargs):
        
        kwargs['kernel_initializer'] = kernel_initializer
        super(SparseConv2D, self).__init__(kernel_size, **kwargs)
        
        self.filters = filters
        self.binary = binary
    
    def build(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape
        
        mask_kernel_shape = (*self.kernel_size, 1, 1)
        mask_fanin = np.prod(mask_kernel_shape[:3])
        # Note: the authors of the paper initialize the mask kernel with ones
        self.mask_kernel = tf.ones(mask_kernel_shape) / tf.cast(mask_fanin, 'float32')
        
        self.kernel_shape = (*self.kernel_size, feature_shape[-1], self.filters)
        
        super(SparseConv2D, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            features = inputs[0]
            mask = inputs[1]
        else:
            # if no mask is provided, get it from the features
            features = inputs
            mask = tf.where(tf.equal(tf.reduce_sum(features, axis=-1, keepdims=True), 0), 0.0, 1.0)

        kernel = self.kernel
        
        if self.weightnorm:
            norm = tf.sqrt(tf.reduce_sum(tf.square(kernel), (0,1,2)) + self.eps)
            kernel = kernel / norm * self.wn_g

        if self.equalize:
            kernel = kernel * self.scale

        features = tf.multiply(features, mask)
        features = K.conv2d(features, kernel,
                            strides=self.strides,
                            padding=self.padding,
                            dilation_rate=self.dilation_rate)
        
        norm = K.conv2d(mask, self.mask_kernel,
                        strides=self.strides,
                        padding=self.padding,
                        dilation_rate=self.dilation_rate)
        
        features = tf.math.divide_no_nan(features, norm)

        if self.use_bias:
            features = tf.add(features, self.bias)
        
        if self.activation is not None:
            features = self.activation(features)
        
        if self.binary:
            mask = tf.where(tf.greater(norm, self.eps), 1.0, 0.0)
        else:
            mask = norm

        return [features, mask]
    
    def compute_output_shape(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape
        
        space = feature_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        
        feature_shape = [feature_shape[0], *new_space, self.filters]
        mask_shape = [feature_shape[0], *new_space, 1]
        
        return [feature_shape, mask_shape]

    def get_config(self):
        config = super(SparseConv2D, self).get_config()
        config.update({
            'filters': self.filters,
            'binary': self.binary,
        })
        return config


class PartialConv2D(Conv2DBaseLayer):
    """2D Partial Convolution layer for sparse input data.
        
    # Arguments
        They are the same as for the normal Conv2D layer.
        binary: Boolean flag, whether the sparsity is propagated as binary 
            mask or as float values.
        weightnorm: Boolean flag, whether Weight Normalization is used or not.
        equalize: Boolean flag, wehter Equalized Learning Rates are used.
    
    # Input shape
        features: 4D tensor with shape (batch_size, rows, cols, channels)
        mask: 4D tensor with shape (batch_size, rows, cols, channels)
            If the shape is (batch_size, rows, cols, 1), the mask is repeated 
            for each channel. If no mask is provided, all input elements 
            unequal to zero are considered as valid.
    
    # Example
        x, m = PartialConv2D(32, 3, padding='same')(x)
        x = Activation('relu')(x)
        x, m = PartialConv2D(32, 3, padding='same')([x,m])
        x = Activation('relu')(x)
    
    # Notes
        In contrast to Sparse Convolution, Partial Convolution propagates 
        the sparsity for each channel separately. This makes it possible 
        to concatenate the features and the masks from different branches 
        in architecture.
    
    # References
        [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723)
        [Sparsity Invariant CNNs](https://arxiv.org/abs/1708.06500)
    """
    def __init__(self, filters, kernel_size, kernel_initializer=conv_init_relu, binary=True, **kwargs):
        
        kwargs['kernel_initializer'] = kernel_initializer
        super(PartialConv2D, self).__init__(kernel_size, **kwargs)
        
        self.filters = filters
        self.binary = binary
    
    def build(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
            self.mask_shape = input_shape[1]
        else:
            feature_shape = input_shape
            self.mask_shape = feature_shape
        
        self.kernel_shape = (*self.kernel_size, feature_shape[-1], self.filters)

        mask_kernel_shape = (*self.kernel_size, feature_shape[-1], self.filters)
        mask_fanin = np.prod(mask_kernel_shape[:3])
        self.mask_kernel = tf.ones(mask_kernel_shape) / tf.cast(mask_fanin, 'float32')
        
        super(PartialConv2D, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            features = inputs[0]
            mask = inputs[1]
            # if mask has only one channel, repeat
            if self.mask_shape[-1] == 1:
                mask = tf.repeat(mask, tf.shape(features)[-1], axis=-1)
        else:
            # if no mask is provided, get it from the features
            features = inputs
            mask = tf.where(tf.equal(features, 0), 0.0, 1.0)
        
        kernel = self.kernel
        
        if self.weightnorm:
            norm = tf.sqrt(tf.reduce_sum(tf.square(kernel), (0,1,2)) + self.eps)
            kernel = kernel / norm * self.wn_g

        if self.equalize:
            kernel = kernel * self.scale

        features = tf.multiply(features, mask)
        features = K.conv2d(features, kernel,
                            strides=self.strides,
                            padding=self.padding,
                            dilation_rate=self.dilation_rate)
        
        norm = K.conv2d(mask, self.mask_kernel,
                        strides=self.strides,
                        padding=self.padding,
                        dilation_rate=self.dilation_rate)
        
        features = tf.math.divide_no_nan(features, norm)
        
        if self.use_bias:
            features = tf.add(features, self.bias)
        
        if self.activation is not None:
            features = self.activation(features)

        if self.binary:
            mask = tf.where(tf.greater(norm, self.eps), 1.0, 0.0)
        else:
            mask = norm

        return [features, mask]
    
    def compute_output_shape(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape
        
        space = feature_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        
        feature_shape = [feature_shape[0], *new_space, self.filters]
        mask_shape = [feature_shape[0], *new_space, self.filters]
        
        return [feature_shape, mask_shape]

    def get_config(self):
        config = super(PartialConv2D, self).get_config()
        config.update({
            'filters': self.filters,
            'binary': self.binary,
        })
        return config


class PartialDepthwiseConv2D(Conv2DBaseLayer):
    """see PartialConv2D and DepthwiseConv2D
    """
    def __init__(self, depth_multiplier, kernel_size, kernel_initializer=depthwiseconv_init_relu, binary=True, **kwargs):
        
        self.depth_multiplier = depth_multiplier

        kwargs['kernel_initializer'] = kernel_initializer
        super(PartialDepthwiseConv2D, self).__init__(kernel_size, **kwargs)
        
        self.binary = binary
    
    def build(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
            self.mask_shape = input_shape[1]
        else:
            feature_shape = input_shape
            self.mask_shape = feature_shape
        
        self.filters = feature_shape[-1] * self.depth_multiplier
        self.kernel_shape = (*self.kernel_size, feature_shape[-1], self.depth_multiplier)

        mask_kernel_shape = (*self.kernel_size, feature_shape[-1], self.depth_multiplier)
        mask_fanin = np.prod(mask_kernel_shape[:2])
        self.mask_kernel = tf.ones(mask_kernel_shape) / tf.cast(mask_fanin, 'float32')

        super(PartialDepthwiseConv2D, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            features = inputs[0]
            mask = inputs[1]
            # if mask has only one channel, repeat
            if self.mask_shape[-1] == 1:
                mask = tf.repeat(mask, tf.shape(features)[-1], axis=-1)
        else:
            # if no mask is provided, get it from the features
            features = inputs
            mask = tf.where(tf.equal(features, 0), 0.0, 1.0)
        
        kernel = self.kernel

        if self.weightnorm:
            norm = tf.sqrt(tf.reduce_sum(tf.square(kernel), (0,1)) + self.eps)
            kernel = kernel / norm * tf.reshape(self.wn_g, (-1, self.depth_multiplier))

        if self.equalize:
            kernel = kernel * self.scale

        features = tf.multiply(features, mask)
        features = K.depthwise_conv2d(features, kernel,
                                      strides=self.strides,
                                      padding=self.padding,
                                      dilation_rate=self.dilation_rate)
        
        norm = K.depthwise_conv2d(mask, self.mask_kernel,
                                  strides=self.strides,
                                  padding=self.padding,
                                  dilation_rate=self.dilation_rate)
        
        features = tf.math.divide_no_nan(features, norm)
        
        if self.use_bias:
            features = tf.add(features, self.bias)
        
        if self.activation is not None:
            features = self.activation(features)

        if self.binary:
            mask = tf.where(tf.greater(norm, self.eps), 1.0, 0.0)
        else:
            mask = norm
        
        return [features, mask]
    
    def compute_output_shape(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape
        
        space = feature_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        
        feature_shape = [feature_shape[0], *new_space, feature_shape[-1]*self.depth_multiplier]
        mask_shape = [feature_shape[0], *new_space, feature_shape[-1]*self.depth_multiplier]
        
        return [feature_shape, mask_shape]
    
    def get_config(self):
        config = super(PartialDepthwiseConv2D, self).get_config()
        config.update({
            'depth_multiplier': self.depth_multiplier,
            'binary': self.binary,
        })
        return config


class GroupConv2D(Conv2DBaseLayer):
    """2D Group Convolution layer that shares weights over symmetries.
    
    Group Convolution provides discrete rotation equivariance. It reduces the number 
    of parameters and typically lead to better results.
    
    The following two finite groups are supported:
        Cyclic Group C4 (p4, 4 rotational symmetries)
        Dihedral Group D4 (p4m, 4 rotational and 4 reflection symmetries)
    
    # Arguments
        They are the same as for the normal Conv2D layer.
        filters: int, The effective number of filters is this value multiplied by the
            number of transformations in the group (4 for C4 and 8 for D4)
        kernel_size: int, Only odd values are supported
        group: 'C4' or 'D4', Stay with one group when stacking layers
        
    # Input shape
        featurs: 4D tensor with shape (batch_size, rows, cols, in_channels)
            or 5D tensor with shape (batch_size, rows, cols, num_transformations, in_channels)
    
    # Output shape
        featurs: 5D tensor with shape (batch_size, rows, cols, num_transformations, out_channels)
    
    # Notes
        - BatchNormalization works as expected and shares the statistict over symmetries.
        - Spatial Pooling can be done via AvgPool3D.
        - Pooling along the group dimension can be done via MaxPool3D.
        - Concatenation along the group dimension can be done via Reshape.
        - To get a model with the inference time of a normal CNN, you can load the
          expanded kernel into a normal Conv2D layer. The kernel expansion is
          done in the 'call' method and the expanded kernel is stored in the
          'transformed_kernel' attribute.
    
    # Example
        x = Input((16,16,3))
        x = GroupConv2D(12, 3, group='D4', padding='same', activation='relu')(x)
        x = BatchNormalization()(x)
        x = GroupConv2D(12, 3, group='D4', padding='same', activation='relu')(x)
        x = AvgPool3D(pool_size=(2,2,1), strides=(2,2,1), padding='same')(x)
        x = GroupConv2D(12, 3, group='D4', padding='same', activation='relu')(x)
        x = MaxPool3D(pool_size=(1,1,x.shape[-2]))(x)
        s = x.shape
        x = Reshape((s[1],s[2],s[3]*s[4]))(x)
        
    # References
        [Group Equivariant Convolutional Networks](https://arxiv.org/abs/1602.07576)
        [Rotation Equivariant CNNs for Digital Pathology](https://arxiv.org/abs/1806.03962)
        
        https://github.com/tscohen/GrouPy
        https://github.com/basveeling/keras-gcnn
    """
    
    def __init__(self, filters, kernel_size, group='D4', **kwargs):
        super(GroupConv2D, self).__init__(kernel_size, **kwargs)
        
        if not self.kernel_size[0] == self.kernel_size[1]:
            raise ValueError('Requires square kernel')
        if self.kernel_size[0] % 2 != 1:
            raise ValueError('Requires odd kernel size')
        
        group = group.upper()
        if group == 'C4':
            self.num_transformations = 4
        elif group == 'D4':
            self.num_transformations = 8
        else:
            raise ValueError('Unknown group')
        
        self.filters = filters
        self.group = group
        
        self.input_spec = InputSpec(min_ndim=4, max_ndim=5)
    
    def compute_output_shape(self, input_shape):
        space = input_shape[1:3]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)
        return (input_shape[0], *new_space, self.num_transformations, self.filters)
    
    def build(self, input_shape):
        
        if len(input_shape) == 4:
            self.first = True
            num_in_channels = input_shape[-1]
        else:
            self.first = False
            num_in_channels = input_shape[-2] * input_shape[-1]
        
        self.kernel_shape = (*self.kernel_size, num_in_channels, self.filters)

        self.kernel = self.add_weight(name='kernel',
                                      shape=self.kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True,
                                      dtype=self.dtype)
        
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True,
                                        dtype=self.dtype)
        else:
            self.bias = None
        
        #super(GroupConv2D, self).build(input_shape)
        super(Conv2DBaseLayer, self).build(input_shape)
    
    def call(self, features):
        ni = features.shape[-1]
        no = self.filters
        
        if self.group == 'C4':
            nt = 4
        elif self.group == 'D4':
            nt = 8
            
        nti = 1 if self.first else nt
        nto = nt
        
        k = self.kernel_size[0]
        t = np.reshape(np.arange(nti*k*k), (nti,k,k))
        trafos = [np.rot90(t,k,axes=(1, 2)) for k in range(4)]
        if nt == 8:
            trafos = trafos + [np.flip(t,1) for t in trafos]
        self.trafos = trafos = np.array(trafos)
        
        # index magic happens here
        if nti == 1:
            indices = trafos
        elif nti == 4:
            indices = [[trafos[l, (m-l)%4 ,:,:] for m in range(4)] for l in range(4)]
        elif nti == 8:
            indices = [[trafos[l, (m-l)%4 if ((m < 4) == (l < 4)) else (m+l)%4+4 ,:,:] for m in range(8)] for l in range(8)]
        self.indices = indices = np.reshape(indices, (nto,nti,k,k))
        
        # transform the kernel
        kernel = self.kernel
        kernel = tf.reshape(kernel, (nti*k*k, ni, no))
        kernel = tf.gather(kernel, indices, axis=0)
        kernel = tf.reshape(kernel, (nto, nti, k,k, ni, no))
        kernel = tf.transpose(kernel, (2,3,1,4,0,5))
        kernel = tf.reshape(kernel, (k,k, nti*ni, nto*no))
        self.transformed_kernel = kernel
        
        if self.first:
            x = features
        else:
            s = features.shape
            x = tf.reshape(features, (-1,s[1],s[2],s[3]*s[4]))
        
        x = K.conv2d(x, kernel, strides=self.strides, padding=self.padding, dilation_rate=self.dilation_rate)
        s = x.shape
        x = tf.reshape(x, (-1,s[1],s[2],nto,no))
        features = x
        
        if self.use_bias:
            features = tf.add(features, self.bias)
        
        if self.activation is not None:
            features = self.activation(features)
        
        return features
    
    def get_config(self):
        config = super(GroupConv2D, self).get_config()
        config.update({
            'filters': self.filters,
            'group': self.group,
        })
        return config


class DeformableConv2D(Conv2DBaseLayer):
    """2D Deformable Convolution layer that learns the spatial offsets where 
    the input elements of the convolution are sampled.
    
    The layer is basically a updated version of An Jiaoyang's code.
    
    # Notes
        - The layer does not use a native CUDA kernel which would have better 
          performance https://github.com/tensorflow/addons/issues/179
    
    # References
        [Deformable Convolutional Networks](https://arxiv.org/abs/1703.06211)
    
    # related code
        https://github.com/DHZS/tf-deformable-conv-layer (An Jiaoyang, 2018-10-11)
    """
    
    def __init__(self, filters, kernel_size, num_deformable_group=None, **kwargs):
        """`kernel_size`, `strides` and `dilation_rate` must have the same value in both axis.
        
        :param num_deformable_group: split output channels into groups, offset shared in each group. If
        this parameter is None, then set num_deformable_group=filters.
        """
        super(DeformableConv2D, self).__init__(kernel_size, **kwargs)
        
        if not self.kernel_size[0] == self.kernel_size[1]:
            raise ValueError('Requires square kernel')
        if not self.strides[0] == self.strides[1]:
            raise ValueError('Requires equal stride')
        if not self.dilation_rate[0] == self.dilation_rate[1]:
            raise ValueError('Requires equal dilation')
        
        self.filters = filters
        if num_deformable_group is None:
            num_deformable_group = filters
        if filters % num_deformable_group != 0:
            raise ValueError('"filters" mod "num_deformable_group" must be zero')
        self.num_deformable_group = num_deformable_group
        
        self.kernel = None
        self.bias = None
        self.offset_layer_kernel = None
        self.offset_layer_bias = None
    
    def build(self, input_shape):
        
        input_dim = input_shape[-1]
        
        # create offset conv layer
        offset_num = self.kernel_size[0] * self.kernel_size[1] * self.num_deformable_group
        self.offset_layer_kernel = self.add_weight(name='offset_layer_kernel',
                        shape=self.kernel_size + (input_dim, offset_num * 2),  # 2 means x and y axis
                        initializer=tf.zeros_initializer(),
                        regularizer=self.kernel_regularizer,
                        trainable=True,
                        dtype=self.dtype)
        self.offset_layer_bias = self.add_weight(name='offset_layer_bias',
                        shape=(offset_num * 2,),
                        initializer=tf.zeros_initializer(),
                        # initializer=tf.random_uniform_initializer(-5, 5),
                        regularizer=self.bias_regularizer,
                        trainable=True,
                        dtype=self.dtype)

        # kernel_shape = self.kernel_size + (input_dim, self.filters)
        # we want to use depth-wise conv
        self.kernel_shape = self.kernel_size + (self.filters * input_dim, 1)

        self.kernel = self.add_weight(name='kernel',
                        shape=self.kernel_shape,
                        initializer=self.kernel_initializer,
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint,
                        trainable=True,
                        dtype=self.dtype)
        
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                            shape=(self.filters,),
                            initializer=self.bias_initializer,
                            regularizer=self.bias_regularizer,
                            constraint=self.bias_constraint,
                            trainable=True,
                            dtype=self.dtype)
        
        #super(DeformableConv2D, self).build(input_shape)
        super(Conv2DBaseLayer, self).build(input_shape)
    
    def call(self, inputs, training=None, **kwargs):
        # get offset, shape [batch_size, out_h, out_w, filter_h, * filter_w * channel_out * 2]
        offset = tf.nn.conv2d(inputs,
                              filters=self.offset_layer_kernel,
                              strides=[1, *self.strides, 1],
                              padding=self.padding.upper(),
                              dilations=[1, *self.dilation_rate, 1])
        offset += self.offset_layer_bias
        
        # add padding if needed
        inputs = self._pad_input(inputs)
        
        # some length
        batch_size = tf.shape(inputs)[0]
        channel_in = int(inputs.shape[-1])
        
        in_h, in_w = [int(i) for i in inputs.shape[1: 3]]  # input feature map size
        out_h, out_w = [int(i) for i in offset.shape[1: 3]]  # output feature map size
        filter_h, filter_w = self.kernel_size
        
        # get x, y axis offset
        offset = tf.reshape(offset, [batch_size, out_h, out_w, -1, 2])
        y_off, x_off = offset[:, :, :, :, 0], offset[:, :, :, :, 1]
        
        # input feature map gird coordinates
        y, x = self._get_conv_indices([in_h, in_w])
        y, x = [tf.expand_dims(i, axis=-1) for i in [y, x]]
        y, x = [tf.tile(i, [batch_size, 1, 1, 1, self.num_deformable_group]) for i in [y, x]]
        y, x = [tf.reshape(i, [batch_size, *i.shape[1: 3], -1]) for i in [y, x]]
        y, x = [tf.cast(i, 'float32') for i in [y, x]]
        
        # add offset
        y, x = y + y_off, x + x_off
        y = tf.clip_by_value(y, 0, in_h - 1)
        x = tf.clip_by_value(x, 0, in_w - 1)
        
        # get four coordinates of points around (x, y)
        y0, x0 = [tf.cast(tf.floor(i), 'int32') for i in [y, x]]
        y1, x1 = y0 + 1, x0 + 1
        # clip
        y0, y1 = [tf.clip_by_value(i, 0, in_h - 1) for i in [y0, y1]]
        x0, x1 = [tf.clip_by_value(i, 0, in_w - 1) for i in [x0, x1]]
        
        # get pixel values
        indices = [[y0, x0], [y0, x1], [y1, x0], [y1, x1]]
        p0, p1, p2, p3 = [DeformableConv2D._get_pixel_values_at_point(inputs, i) for i in indices]
        
        # cast to float
        x0, x1, y0, y1 = [tf.cast(i, 'float32') for i in [x0, x1, y0, y1]]
        # weights
        w0 = (y1 - y) * (x1 - x)
        w1 = (y1 - y) * (x - x0)
        w2 = (y - y0) * (x1 - x)
        w3 = (y - y0) * (x - x0)
        # expand dim for broadcast
        w0, w1, w2, w3 = [tf.expand_dims(i, axis=-1) for i in [w0, w1, w2, w3]]
        # bilinear interpolation
        pixels = tf.add_n([w0 * p0, w1 * p1, w2 * p2, w3 * p3])
        
        # reshape the "big" feature map
        pixels = tf.reshape(pixels, [batch_size, out_h, out_w, filter_h, filter_w, self.num_deformable_group, channel_in])
        pixels = tf.transpose(pixels, [0, 1, 3, 2, 4, 5, 6])
        pixels = tf.reshape(pixels, [batch_size, out_h * filter_h, out_w * filter_w, self.num_deformable_group, channel_in])
        
        # copy channels to same group
        feat_in_group = self.filters // self.num_deformable_group
        pixels = tf.tile(pixels, [1, 1, 1, 1, feat_in_group])
        pixels = tf.reshape(pixels, [batch_size, out_h * filter_h, out_w * filter_w, -1])
        
        # depth-wise conv
        out = tf.nn.depthwise_conv2d(pixels, self.kernel, [1, filter_h, filter_w, 1], 'VALID')
        # add the output feature maps in the same group
        out = tf.reshape(out, [batch_size, out_h, out_w, self.filters, channel_in])
        out = tf.reduce_sum(out, axis=-1)

        features = out

        if self.use_bias:
            features = tf.add(features, self.bias)
        
        if self.activation is not None:
            features = self.activation(features)

        return features
    
    def _pad_input(self, inputs):
        """Check if input feature map needs padding, because we don't use the standard Conv() function.
        
        :param inputs:
        :return: padded input feature map
        """
        # When padding is 'same', we should pad the feature map.
        # if padding == 'same', output size should be `ceil(input / stride)`
        if self.padding == 'same':
            in_shape = inputs.shape.as_list()[1:3]
            padding_list = []
            for i in range(2):
                filter_size = self.kernel_size[i]
                dilation = self.dilation_rate[i]
                dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
                same_output = (in_shape[i] + self.strides[i] - 1) // self.strides[i]
                valid_output = (in_shape[i] - dilated_filter_size + self.strides[i]) // self.strides[i]
                if same_output == valid_output:
                    padding_list += [0, 0]
                else:
                    p = dilated_filter_size - 1
                    p_0 = p // 2
                    padding_list += [p_0, p - p_0]
            if sum(padding_list) != 0:
                padding = [[0, 0],
                           [padding_list[0], padding_list[1]],  # top, bottom padding
                           [padding_list[2], padding_list[3]],  # left, right padding
                           [0, 0]]
                inputs = tf.pad(inputs, padding)
        return inputs
    
    def _get_conv_indices(self, feature_map_size):
        """the x, y coordinates in the window when a filter sliding on the feature map

        :param feature_map_size:
        :return: y, x with shape [1, out_h, out_w, filter_h * filter_w]
        """
        feat_h, feat_w = [int(i) for i in feature_map_size[0: 2]]

        x, y = tf.meshgrid(tf.range(feat_w), tf.range(feat_h))
        x, y = [tf.reshape(i, [1, *i.get_shape(), 1]) for i in [x, y]]  # shape [1, h, w, 1]
        x, y = [tf.image.extract_patches(i,
                                         [1, *self.kernel_size, 1],
                                         [1, *self.strides, 1],
                                         [1, *self.dilation_rate, 1],
                                         'VALID')
                for i in [x, y]]  # shape [1, out_h, out_w, filter_h * filter_w]
        return y, x

    @staticmethod
    def _get_pixel_values_at_point(inputs, indices):
        """get pixel values

        :param inputs:
        :param indices: shape [batch_size, H, W, I], I = filter_h * filter_w * channel_out
        :return:
        """
        y, x = indices
        batch, h, w, n = y.shape.as_list()[0: 4]
        
        y_shape = tf.shape(y)
        batch, n = y_shape[0], y_shape[3]
        
        batch_idx = tf.reshape(tf.range(0, batch), (batch, 1, 1, 1))
        b = tf.tile(batch_idx, (1, h, w, n))
        pixel_idx = tf.stack([b, y, x], axis=-1)
        return tf.gather_nd(inputs, pixel_idx)


class MaxPoolingWithArgmax2D(Layer):
    '''MaxPooling for unpooling with indices.
    
    # References
        [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](http://arxiv.org/abs/1511.00561)
    
    # related code:
        https://github.com/PavlosMelissinos/enet-keras
        https://github.com/ykamikawa/SegNet
    '''
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='same', **kwargs):
        super(MaxPoolingWithArgmax2D, self).__init__(**kwargs)
        self.pool_size = conv_utils.normalize_tuple(pool_size, 2, 'pool_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)

    def call(self, inputs, **kwargs):
        size = (1, self.pool_size[0], self.pool_size[1], 1)
        strides = (1, self.strides[0], self.strides[1], 1)
        padding = self.padding.upper()
        output, idxs = tf.nn.max_pool_with_argmax(inputs, size, strides, padding)
        return [output, idxs]
    
    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0], input_shape[1]//self.strides[0], input_shape[2]//self.strides[1], input_shape[3])
        return [output_shape, output_shape]
    
    def get_config(self):
        config = super(MaxPoolingWithArgmax2D, self).get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding,
        })
        return config


class MaxUnpooling2D(Layer):
    '''Inversion of MaxPooling with indices.
    
    # References
        [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](http://arxiv.org/abs/1511.00561)
    
    # related code:
        https://github.com/PavlosMelissinos/enet-keras
        https://github.com/ykamikawa/SegNet
    '''
    def __init__(self, size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')

    def call(self, inputs):
        features, idxs = inputs[0], inputs[1]
        
        input_shape = tf.shape(features, out_type='int64')
        output_shape = (input_shape[0], input_shape[1]*self.size[0], input_shape[2]*self.size[1], input_shape[3])
        
        # calculation indices for batch, height, width and feature maps
        idxs = tf.reshape(idxs, (-1,))
        b = tf.repeat(tf.range(input_shape[0]), (input_shape[1]*input_shape[2]*input_shape[3],))
        y = idxs // (output_shape[2] * output_shape[3])
        x = (idxs // output_shape[3]) % output_shape[2]
        c = tf.tile(tf.range(input_shape[3]), (input_shape[0]*input_shape[1]*input_shape[2],))
        
        # transpose indices & reshape update values to one dimension
        indices = K.stack([b,y,x,c], axis=-1)
        updates = K.reshape(features, (-1,))
        features = tf.scatter_nd(indices, updates, output_shape)
        return features
    
    def compute_output_shape(self, input_shape):
        idxs_shape = input_shape[1]
        return (idxs_shape[0], idxs_shape[1]*self.size[0], idxs_shape[2]*self.size[1], idxs_shape[3])
    
    def get_config(self):
        config = super(MaxUnpooling2D, self).get_config()
        config.update({
            'size': self.size,
        })
        return config


class AddCoords2D(Layer):
    """Add coords to a tensor as described in CoordConv paper.

    # Arguments
        with_r: Boolean flag, whether the r coordinate is added or not. See paper for more details.
    
    # Input shape
        featurs: 4D tensor with shape (batch_size, rows, cols, channels)

    # Output shape
        featurs: same as input except channels + 2, channels + 3 if with_r is True
    
    # Example
        x = Conv2D(32, 3, padding='same', activation='relu')(x)
        x = AddCoords2D()(x)
        x = Conv2D(32, 3, padding='same', activation='relu')(x)
    
    # Notes
        Semi-convolutional Operators is an approach that is closely related to CoordConv.
    
    # References
        [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](http://arxiv.org/abs/1807.03247)
        [Semi-convolutional Operators for Instance Segmentation](https://arxiv.org/abs/1807.10712)
    """
    def __init__(self, with_r=False, **kwargs):
        super(AddCoords2D, self).__init__(**kwargs)
        self.with_r = with_r
        
    def call(self, features):
        y_dim = features.shape[1]
        x_dim = features.shape[2]
        
        ones = tf.ones_like(features[:,:,:,:1])
        y_range = tf.range(y_dim, dtype='float32') / tf.cast(y_dim-1, 'float32') * 2 - 1
        x_range = tf.range(x_dim, dtype='float32') / tf.cast(x_dim-1, 'float32') * 2 - 1
        yy = ones * y_range[None, :, None, None]
        xx = ones * x_range[None, None, :, None]
        
        if self.with_r:
            rr = tf.sqrt(tf.square(yy-0.5) + tf.square(xx-0.5))
            features = tf.concat([features, yy, xx, rr], axis=-1)
        else:
            features = tf.concat([features, yy, xx], axis=-1)
        return features
    
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[3] = output_shape[3] + 2
        if self.with_r:
            output_shape[3] = output_shape[3] + 1
        return tuple(output_shape)
    
    def get_config(self):
        config = super(AddCoords2D, self).get_config()
        config.update({
            'with_r': self.with_r,
        })
        return config


class LayerNormalization(Layer):
    """Layer Normalization Layer.
    
    # References
        [Layer Normalization](http://arxiv.org/abs/1607.06450)
    """
    def __init__(self, eps=1e-6, **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.eps = eps
    
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=initializers.Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=initializers.Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    
    def call(self, x):
        mean = tf.stop_gradient(K.mean(x, axis=-1, keepdims=True))
        std = tf.stop_gradient(K.std(x, axis=-1, keepdims=True))
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(LayerNormalization, self).get_config()
        config.update({
            'eps': self.eps,
        })
        return config


class InstanceNormalization(Layer):
    """Instance Normalization Layer.
    
    # References
        [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
    """
    def __init__(self, eps=1e-6, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.eps = eps
    
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=initializers.Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=initializers.Zeros(), trainable=True)
        super(InstanceNormalization, self).build(input_shape)
    
    def call(self, x):
        axis = list(range(len(x.shape))[1:-1])
        mean = tf.stop_gradient(K.mean(x, axis=axis, keepdims=True))
        std = tf.stop_gradient(K.std(x, axis=axis, keepdims=True))
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({
            'eps': self.eps,
        })
        return config


def Resize2D(size, method='bilinear'):
    """Spatial resizing layer.
    
    # Arguments
        size: spatial output size (rows, cols)
        method: 'bilinear', 'bicubic', 'nearest', ...
        
    """
    return Lambda(lambda x: tf.image.resize(x, size, method=method))


class Blur2D(Layer):
    """2D Blur Layer as used in Antialiased CNNs for Subsampling.

    # Notes
        The layer handles boundary effects similar to AvgPool2D.

    # References
        [Making Convolutional Networks Shift-Invariant Again](https://arxiv.org/abs/1904.11486)

    # related code
        https://github.com/adobe/antialiased-cnns
        https://github.com/adobe/antialiased-cnns/issues/10
    """
    def __init__(self, filter_size=3, strides=2, padding='valid', **kwargs):
        rank = 2
        self.filter_size = filter_size
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)

        if self.filter_size == 1:
            self.a = np.array([1.,])
        elif self.filter_size == 2:
            self.a = np.array([1., 1.])
        elif self.filter_size == 3:
            self.a = np.array([1., 2., 1.])
        elif self.filter_size == 4:
            self.a = np.array([1., 3., 3., 1.])
        elif self.filter_size == 5:
            self.a = np.array([1., 4., 6., 4., 1.])
        elif self.filter_size == 6:
            self.a = np.array([1., 5., 10., 10., 5., 1.])
        elif self.filter_size == 7:
            self.a = np.array([1., 6., 15., 20., 15., 6., 1.])

        super(Blur2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        feature_shape = input_shape
        space = feature_shape[1:-1]

        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            new_space.append(new_dim)

        feature_shape = [feature_shape[0], *new_space, feature_shape[3]]
        return feature_shape

    def build(self, input_shape):
        k = self.a[:,None] * self.a[None,:]
        k = np.tile(k[:,:,None,None], (1,1,input_shape[-1],1))
        self.kernel = K.constant(k, dtype=K.floatx())

    def call(self, x):
        features = K.depthwise_conv2d(x, self.kernel, strides=self.strides, padding=self.padding)
        # normalize the features
        mask = tf.ones_like(x)
        norm = K.depthwise_conv2d(mask, self.kernel, strides=self.strides, padding=self.padding)
        features = tf.multiply(features, 1./norm)
        return features

    def get_config(self):
        config = super(Blur2D, self).get_config()
        config.update({
            'filter_size': self.filter_size,
            'strides': self.strides,
            'padding': self.padding,
        })
        return config


class Scale(Layer):
    """Layer to learn a affine feature scaling.
    """
    def __init__(self,
                 use_shift=True,
                 use_scale=True,
                 shift_initializer='zeros',
                 shift_regularizer=None,
                 shift_constraint=None,
                 scale_initializer='ones',
                 scale_regularizer=None,
                 scale_constraint=None,
                 **kwargs):
        super(Scale, self).__init__(**kwargs)
        
        self.use_shift = use_shift
        self.use_scale = use_scale
        self.shift_initializer = initializers.get(shift_initializer)
        self.shift_regularizer = regularizers.get(shift_regularizer)
        self.shift_constraint = constraints.get(shift_constraint)
        self.scale_initializer = initializers.get(scale_initializer)
        self.scale_regularizer = regularizers.get(scale_regularizer)
        self.scale_constraint = constraints.get(scale_constraint)

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        if self.use_shift:
            self.shift = self.add_weight(name='shift',
                                         shape=(input_shape[-1],),
                                         initializer=self.shift_initializer,
                                         regularizer=self.shift_regularizer,
                                         constraint=self.shift_constraint,
                                         trainable=True,
                                         dtype=self.dtype)
        else:
            self.shfit = None

        if self.use_scale:
            self.scale = self.add_weight(name='scale',
                                         shape=(input_shape[-1],),
                                         initializer=self.scale_initializer,
                                         regularizer=self.scale_regularizer,
                                         constraint=self.scale_constraint,
                                         trainable=True,
                                         dtype=self.dtype)
        else:
            self.scale = None

        super(Scale, self).build(input_shape)

    def call(self, inputs, **kwargs):
        x = inputs
        if self.use_scale:
            x = tf.multiply(x, self.scale)
        if self.use_shift:
            x = tf.add(x, self.shift)
        return x

    def get_config(self):
        config = super(Scale, self).get_config()
        config.update({
            'use_shift': self.use_shift,
            'use_scale': self.use_scale,
            'shift_initializer': initializers.serialize(self.shift_initializer),
            'shift_regularizer': regularizers.serialize(self.shift_regularizer),
            'shift_constraint': constraints.serialize(self.shift_constraint),
            'scale_initializer': initializers.serialize(self.scale_initializer),
            'scale_regularizer': regularizers.serialize(self.scale_regularizer),
            'scale_constraint': constraints.serialize(self.scale_constraint),
        })
        return config


def Split(n, axis=-1):
    return Lambda(lambda x: tf.split(x, num_or_size_splits=n, axis=axis))
