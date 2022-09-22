"""
SPDX-License-Identifier: MIT
Copyright © 2018 - 2022 Markus Völk
Code was taken from https://github.com/mvoelk/keras_layers
"""

from keras.layers import Activation, BatchNormalization
from keras.layers import AvgPool2D, MaxPool2D
from keras.layers import concatenate, add

from utils.layers import Conv2D, DepthwiseConv2D, PartialConv2D, PartialDepthwiseConv2D, DeformableConv2D


def subsampling_block(x, pooling='avg', pool_size=2):
    if pooling == 'conv':
        x = Conv2D(x.shape[-1], kernel_size=pool_size, strides=pool_size, padding='same', use_bias=True)(x)
    elif pooling == 'dw':
        x = DepthwiseConv2D(1, kernel_size=pool_size, strides=pool_size, padding='same', use_bias=True)(x)
    elif pooling == 'avg':
        x = AvgPool2D(pool_size=pool_size, padding='same')(x)
    elif pooling == 'max':
        x = MaxPool2D(pool_size=pool_size, padding='same')(x)
    return x


def conv_block(x, filters, kernel_size=1, strides=1, dilation_rate=1, padding='same', 
               use_bias=False, activation='relu', weightnorm=False):
    if weightnorm:
        x = Conv2D(filters, kernel_size=kernel_size, strides=strides, dilation_rate=dilation_rate, 
                   padding=padding, activation=activation, use_bias=use_bias, weightnorm=True)(x)
    else:
        x = Conv2D(filters, kernel_size=kernel_size, strides=strides, 
                   dilation_rate=dilation_rate, padding=padding, use_bias=use_bias)(x)
        x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    return x

def partial_conv_block(xm, features, kernel_size=1, strides=1, dilation_rate=1, padding='same', 
                       use_bias=False, binary=True, activation='relu'):
    xm = PartialConv2D(features, kernel_size, strides=strides, dilation_rate=dilation_rate, 
                       padding=padding, weightnorm=True, activation=activation, binary=binary)(xm)
    return xm

def deformable_conv_block(x, filters, kernel_size=5, strides=1, padding='same', num_deformable_group=8, 
                          use_bias=True, activation='relu'):
    x = DeformableConv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, 
                         num_deformable_group=num_deformable_group, use_bias=use_bias)(x)
    x = BatchNormalization()(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x



def mbconv_block(x, repeats=2, kernel_size=3, expansion=4, dilation_rate=1, use_bias=False, activation='relu', weightnorm=False):
    # MBConv / inverted residual block
    c = x.shape[-1]
    for i in range(repeats):
        x1 = x2 = x
        x1 = conv_block(x1, c*expansion, kernel_size=1, use_bias=use_bias, activation=activation, weightnorm=weightnorm)
        x1 = DepthwiseConv2D(depth_multiplier=1, kernel_size=kernel_size, padding='same', 
                dilation_rate=dilation_rate, use_bias=use_bias, weightnorm=False, activation=activation,
                #kernel_initializer='zeros'
                )(x1)
        x1 = conv_block(x1, c, kernel_size=1, use_bias=use_bias, activation=None, weightnorm=weightnorm)
        x = add([x1, x2])
    return x

def partial_mbconv_block(xm, repeats=2, kernel_size=3, expansion=4, dilation_rate=1, use_bias=False, activation='relu', binary=True):
    # MBConv / inverted residual block
    x, m = xm
    c = x.shape[-1]
    for i in range(repeats):
        xm1 = xm2 = xm
        xm1 = PartialConv2D(c*expansion, kernel_size=1, use_bias=use_bias, weightnorm=True, activation=activation, binary=binary)(xm1)
        xm1 = PartialDepthwiseConv2D(depth_multiplier=1, kernel_size=kernel_size, padding='same', 
                dilation_rate=dilation_rate, use_bias=use_bias, weightnorm=False, activation=activation, binary=binary,
                #kernel_initializer='zeros'
                )(xm1)
        xm1 = PartialConv2D(c, kernel_size=1, use_bias=use_bias, weightnorm=True, activation=None, binary=binary)(xm1)
        x1, m1 = xm1
        x2, m2 = xm2
        xm = [add([x1, x2]), add([m1, m2])/2]
    return xm


def dense_block(x, n, k=32, w=4, kernel_size=3, use_bias=False, activation='relu', weightnorm=False):
    # input, repeats, growth_rate, width
    # note: different from paper bn-relu-conv vs conv-bn-relu
    for i in range(n):
        x1 = x2 = x
        x1 = conv_block(x1, int(k*w), kernel_size=1, use_bias=use_bias, activation=activation, weightnorm=weightnorm)
        x1 = conv_block(x1, k, kernel_size=kernel_size, use_bias=use_bias, activation=activation, weightnorm=weightnorm)
        x = concatenate([x1, x2], axis=3)
    return x


def separable_dense_block(x, n, k=32, w=4, kernel_size=3, use_bias=False, activation='relu', weightnorm=False):
    # input, repeats, growth_rate, width
    for i in range(n):
        x1 = x2 = x
        x1 = conv_block(x1, int(k*w), kernel_size=1, use_bias=use_bias, activation=activation, weightnorm=weightnorm)
        x1 = DepthwiseConv2D(depth_multiplier=1, kernel_size=kernel_size, padding='same', use_bias=False)(x1)
        x1 = conv_block(x1, k, kernel_size=1, use_bias=use_bias, activation=activation, weightnorm=weightnorm)
        x = concatenate([x1, x2], axis=3)
    return x

