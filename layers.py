
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, Lambda
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras import initializers, regularizers, constraints, activations
from tensorflow.python.keras.utils import conv_utils


def gaussian_init(shape, dtype=None, partition_info=None):
    v = np.random.randn(*shape)
    v = np.clip(v, -3, +3)
    return K.constant(v, dtype=dtype)

def conv_init_linear(shape, dtype=None, partition_info=None):
    v = np.random.randn(*shape)
    v = np.clip(v, -3, +3)
    fan_in = np.prod(shape[:3])
    v = v / (fan_in**0.5)
    return K.constant(v, dtype=dtype)

def conv_init_relu(shape, dtype=None, partition_info=None):
    v = np.random.randn(*shape)
    v = np.clip(v, -3, +3)
    fan_in = np.prod(shape[:3])
    v = v / (fan_in**0.5) * 2**0.5
    return K.constant(v, dtype=dtype)

def conv_init_relu2(shape, dtype=None, partition_info=None):
    v = np.random.randn(*shape)
    v = np.clip(v, -3, +3)
    fan_in = np.prod(shape[:3])
    v = v / (fan_in**0.5) * 2
    return K.constant(v, dtype=dtype)

def depthwiseconv_init_linear(shape, dtype=None, partition_info=None):
    v = np.random.randn(*shape)
    v = np.clip(v, -3, +3)
    fan_in = np.prod(shape[:2])
    v = v / (fan_in**0.5)
    return K.constant(v, dtype=dtype)

def depthwiseconv_init_relu(shape, dtype=None, partition_info=None):
    v = np.random.randn(*shape)
    v = np.clip(v, -3, +3)
    fan_in = np.prod(shape[:2])
    v = v / (fan_in**0.5) * 2**0.5
    return K.constant(v, dtype=dtype)


class Covn2DBaseLayer(Layer):
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
                 **kwargs):

        super(Covn2DBaseLayer, self).__init__(
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


class Conv2D(Covn2DBaseLayer):
    """Conv2D Layer with Weight Normalization.
    
    # Arguments
        They are the same as for the normal Conv2D layer.
        weightnorm: Boolean flag, whether Weight Normalization is used or not.
        
    # References
        [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](http://arxiv.org/abs/1602.07868)
    """
    def __init__(self, filters, kernel_size, weightnorm=False, eps=1e-6, **kwargs):
        super(Conv2D, self).__init__(kernel_size, **kwargs)
        
        self.filters = filters
        self.weightnorm = weightnorm
        self.eps = eps
    
    def build(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape
        
        self.kernel_shape = (*self.kernel_size, feature_shape[-1], self.filters)
        self.kernel = self.add_weight(name='kernel',
                                      shape=self.kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True,
                                      dtype=self.dtype)
        
        if self.weightnorm:
            self.wn_g = self.add_weight(name='wn_g',
                                        shape=(self.filters,),
                                        initializer=initializers.Ones(),
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
        
        super(Conv2D, self).build(input_shape)
        
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            features = inputs[0]
        else:
            features = inputs
            
        if self.weightnorm:
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.kernel), (0,1,2)) + self.eps)
            kernel = self.kernel / norm * self.wn_g
        else:
            kernel = self.kernel
        
        features = K.conv2d(features, kernel,
                            strides=self.strides,
                            padding=self.padding,
                            dilation_rate=self.dilation_rate)
        
        if self.use_bias:
            features = tf.add(features, self.bias)
        
        if self.activation is not None:
            features = self.activation(features)
        
        return features


class SparseConv2D(Covn2DBaseLayer):
    """2D Sparse Convolution layer for sparse input data.
    
    # Arguments
        They are the same as for the normal Conv2D layer.
        binary: Boolean flag, whether the sparsity is propagated as binary 
            mask or as float values.
    
    # Input shape
        features: 4D tensor with shape (batch_size, rows, cols, channels)
        mask: 4D tensor with shape (batch_size, rows, cols, 1)
            If no mask is provided, all input pixels with features unequal 
            to zero are considered as valid.
    
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
    def __init__(self, filters, kernel_size,
                 kernel_initializer=conv_init_relu,
                 binary=True,
                 **kwargs):
        
        super(SparseConv2D, self).__init__(kernel_size, kernel_initializer=kernel_initializer, **kwargs)
        
        self.filters = filters
        self.binary = binary
    
    def build(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape
        
        self.kernel_shape = (*self.kernel_size, feature_shape[-1], self.filters)
        self.kernel = self.add_weight(name='kernel',
                                      shape=self.kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True,
                                      dtype=self.dtype)
        
        self.mask_kernel_shape = (*self.kernel_size, 1, 1)
        self.mask_kernel = tf.ones(self.mask_kernel_shape)
        self.mask_fan_in = tf.reduce_prod(self.mask_kernel_shape[:3])
        
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
        
        super(SparseConv2D, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            features = inputs[0]
            mask = inputs[1]
        else:
            # if no mask is provided, get it from the features
            features = inputs
            mask = tf.where(tf.equal(tf.reduce_sum(features, axis=-1, keepdims=True), 0), 0.0, 1.0) 
            
        features = tf.multiply(features, mask)
        features = nn_ops.convolution(features, self.kernel, self.padding.upper(), self.strides, self.dilation_rate)

        norm = nn_ops.convolution(mask, self.mask_kernel, self.padding.upper(), self.strides, self.dilation_rate)
        
        mask_fan_in = tf.cast(self.mask_fan_in, 'float32')
        
        if self.binary:
            mask = tf.where(tf.greater(norm,0), 1.0, 0.0)
        else:
            mask = norm / mask_fan_in
        
        #ratio = tf.where(tf.equal(norm,0), 0.0, 1/norm) # Note: The authors use this in the paper, but it would require special initialization...
        ratio = tf.where(tf.equal(norm,0), 0.0, mask_fan_in/norm)
        
        features = tf.multiply(features, ratio)
        
        if self.use_bias:
            features = tf.add(features, self.bias)
        
        if self.activation is not None:
            features = self.activation(features)
        
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
        mask_shape = [*feature_shape[:-1], 1]
        
        return [feature_shape, mask_shape]


class PartialConv2D(Covn2DBaseLayer):
    """2D Partial Convolution layer for sparse input data.
        
    # Arguments
        They are the same as for the normal Conv2D layer.
        binary: Boolean flag, whether the sparsity is propagated as binary 
            mask or as float values.
    
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
    def __init__(self, filters, kernel_size,
                 kernel_initializer=conv_init_relu,
                 binary=True,
                 **kwargs):
        
        super(PartialConv2D, self).__init__(kernel_size, kernel_initializer=kernel_initializer, **kwargs)
        
        self.filters = filters
        self.binary = binary
    
    def build(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
            mask_shape = input_shape[1]
            self.mask_shape = mask_shape
        else:
            feature_shape = input_shape
            self.mask_shape = feature_shape
        
        self.kernel_shape = (*self.kernel_size, feature_shape[-1], self.filters)
        self.kernel = self.add_weight(name='kernel',
                                      shape=self.kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True,
                                      dtype=self.dtype)
        
        self.mask_kernel_shape = (*self.kernel_size, feature_shape[-1], self.filters)
        self.mask_kernel = tf.ones(self.mask_kernel_shape)
        self.mask_fan_in = tf.reduce_prod(self.mask_kernel_shape[:3])
        
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
            
        features = tf.multiply(features, mask)
        features = nn_ops.convolution(features, self.kernel, self.padding.upper(), self.strides, self.dilation_rate)

        norm = nn_ops.convolution(mask, self.mask_kernel, self.padding.upper(), self.strides, self.dilation_rate)
        
        mask_fan_in = tf.cast(self.mask_fan_in, 'float32')
        
        if self.binary:
            mask = tf.where(tf.greater(norm,0), 1.0, 0.0)
        else:
            mask = norm / mask_fan_in
        
        ratio = tf.where(tf.equal(norm,0), 0.0, mask_fan_in/norm)
        
        features = tf.multiply(features, ratio)
        
        if self.use_bias:
            features = tf.add(features, self.bias)
        
        if self.activation is not None:
            features = self.activation(features)
        
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


class DepthwiseConv2D(Covn2DBaseLayer):
    """2D depthwise convolution layer.
    
    # Notes
        A DepthwiseConv2D layer followed by an 1x1 Conv2D layer is equivalent
        to the SeparableConv2D layer provided by Keras.
    
    # References
        [Xception: Deep Learning with Depthwise Separable Convolutions](http://arxiv.org/abs/1610.02357)
    """
    def __init__(self, depth_multiplier, kernel_size,
                 kernel_initializer=depthwiseconv_init_relu,
                 **kwargs):
        super(DepthwiseConv2D, self).__init__(kernel_size, kernel_initializer=kernel_initializer, **kwargs)
        
        self.depth_multiplier = depth_multiplier
        
    def build(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape
        
        kernel_shape = (*self.kernel_size, feature_shape[-1], self.depth_multiplier)
        
        self.kernel = self.add_weight(name='kernel',
                                      shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=True,
                                      dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(feature_shape[-1]*self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint,
                                        trainable=True,
                                        dtype=self.dtype)
        else:
            self.bias = None
        
        super(DepthwiseConv2D, self).build(input_shape)
    
    def call(self, inputs, **kwargs):
        if type(inputs) is list:
            features = inputs[0]
        else:
            features = inputs
        
        features = K.depthwise_conv2d(features, self.kernel,
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
        ksize = [1, self.pool_size[0], self.pool_size[1], 1]
        strides = [1, self.strides[0], self.strides[1], 1]
        padding = self.padding.upper()
        output, argmax = nn_ops.max_pool_with_argmax(inputs, ksize, strides, padding)
        argmax = tf.cast(argmax, K.floatx())
        return [output, argmax]
    
    def compute_output_shape(self, input_shape):
        ratio = (1, 2, 2, 1)
        output_shape = [dim // ratio[idx] if dim is not None else None for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [output_shape, output_shape]

    def compute_mask(self, inputs, mask=None):
        return 2 * [None]


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

    def call(self, inputs, output_shape=None):
        updates, mask = inputs[0], inputs[1]
        
        mask = tf.cast(mask, 'int32')
        input_shape = tf.shape(updates, out_type='int32')
        #  calculation new shape
        if output_shape is None:
            output_shape = (input_shape[0], input_shape[1] * self.size[0], input_shape[2] * self.size[1], input_shape[3])
        
        # calculation indices for batch, height, width and feature maps
        one_like_mask = K.ones_like(mask, dtype='int32')
        batch_shape = K.concatenate([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = K.reshape(tf.range(output_shape[0], dtype='int32'), shape=batch_shape)
        b = one_like_mask * batch_range
        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]
        feature_range = tf.range(output_shape[3], dtype='int32')
        f = one_like_mask * feature_range
        
        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(updates)
        indices = K.transpose(K.reshape(K.stack([b, y, x, f]), [4, updates_size]))
        values = K.reshape(updates, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret
    
    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        output_shape = [mask_shape[0], mask_shape[1] * self.size[0], mask_shape[2] * self.size[1], mask_shape[3]]
        return tuple(output_shape)


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
    
    # References
        [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](http://arxiv.org/abs/1807.03247)
    """
    def __init__(self, with_r=False, **kwargs):
        super(AddCoords2D, self).__init__(**kwargs)
        self.with_r = with_r
        
    def call(self, input_tensor):
        input_shape = tf.shape(input_tensor)
        batch_size = input_shape[0]
        x_dim = input_shape[1]
        y_dim = input_shape[2]
        
        xx_ones = tf.ones([batch_size, x_dim], dtype=tf.int32)
        xx_ones = tf.expand_dims(xx_ones, -1)
        xx_range = tf.tile(tf.expand_dims(tf.range(x_dim), 0), [batch_size, 1])
        xx_range = tf.expand_dims(xx_range, 1)
        xx_channel = tf.matmul(xx_ones, xx_range)
        xx_channel = tf.expand_dims(xx_channel, -1)
        xx_channel = tf.cast(xx_channel, 'float32') / (tf.cast(x_dim, 'float32') - 1)
        xx_channel = xx_channel*2 - 1
        
        yy_ones = tf.ones([batch_size, y_dim], dtype=tf.int32)
        yy_ones = tf.expand_dims(yy_ones, 1)
        yy_range = tf.tile(tf.expand_dims(tf.range(y_dim), 0), [batch_size, 1])
        yy_range = tf.expand_dims(yy_range, -1)
        yy_channel = tf.matmul(yy_range, yy_ones)
        yy_channel = tf.expand_dims(yy_channel, -1)
        yy_channel = tf.cast(yy_channel, 'float32') / (tf.cast(x_dim, 'float32') - 1)
        yy_channel = yy_channel*2 - 1
        
        output_tensor = tf.concat([input_tensor, xx_channel, yy_channel], axis=-1)
        if self.with_r:
            rr = tf.sqrt(tf.square(xx_channel-0.5) + tf.square(yy_channel-0.5))
            output_tensor = tf.concat([output_tensor, rr], axis=-1)
        return output_tensor
    
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[3] = output_shape[3] + 2
        if self.with_r:
            output_shape[3] = output_shape[3] + 1
        return tuple(output_shape)


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
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape


def Resize2D(size, method='bilinear'):
    """Spatial resizing layer.
    
    # Arguments
        size: spatial output size (rows, cols)
        method: 'bilinear', 'bicubic', 'nearest', ...
        
    """
    return Lambda(lambda x: tf.image.resize(x, size, method=method))


class Blur2D(Layer):
    """2D Blur Layer as used in Antialiased CNNs for Subsampling

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

