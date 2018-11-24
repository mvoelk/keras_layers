
import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer, Lambda
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras import initializers, regularizers, constraints
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils


class SparseConv2D(Layer):
    """2D sparse convolution layer for sparse input data.
    
    # Example
        x, m = SparseConv2D(32, 3, padding='same')(x)
        x = Activation('relu')(x)
        x, m = SparseConv2D(32, 3, padding='same')([x,m])
        x = Activation('relu')(x)
    
    # Notes
        Sparse Convolution is the same idea as Partial Convolution.
    
    # References
        [Sparsity Invariant CNNs](https://arxiv.org/abs/1708.06500)  
        [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723)
    """
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 #data_format=None,
                 dilation_rate=(1, 1),
                 #activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 #activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 binary=True,
                 **kwargs):
        super(SparseConv2D, self).__init__(**kwargs)
        
        rank = 2
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.binary = binary
    
    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape
        
        kernel_shape = [*self.kernel_size, feature_shape[-1], self.filters]
        
        self.kernel = self.add_variable(name='kernel',
                                        shape=kernel_shape,
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        constraint=self.kernel_constraint,
                                        trainable=True,
                                        dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_variable(name='bias',
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
            # if no maks is provided, get it from the features
            features = inputs
            mask = tf.expand_dims(tf.reduce_sum(features, axis=-1), axis=-1)
            mask = tf.where(tf.equal(mask, 0), tf.zeros_like(mask), tf.ones_like(mask)) 
            
        features = tf.multiply(features, mask)
        features = nn_ops.convolution(features, self.kernel, self.padding.upper(), self.strides, self.dilation_rate)

        kernel = tf.ones([*self.kernel_size, 1, 1])
        norm = nn_ops.convolution(mask, kernel, self.padding.upper(), self.strides, self.dilation_rate)
        
        if self.binary:
            mask = nn_ops.pool(mask, self.kernel_size, 'MAX', self.padding.upper(), self.dilation_rate, self.strides)
        else:
            mask = norm / np.prod(self.kernel_size)
        
        norm = tf.where(tf.equal(norm,0), tf.zeros_like(norm), tf.reciprocal(norm))
        
        features = tf.multiply(features, norm)
        if self.use_bias:
            features = tf.add(features, self.bias)
        
        return [features, mask]
    
    @tf_utils.shape_type_conversion
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
    
    @tf_utils.shape_type_conversion
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
        with tf.variable_scope(self.name):
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
    
    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        mask_shape = input_shape[1]
        output_shape = [mask_shape[0], mask_shape[1] * self.size[0], mask_shape[2] * self.size[1], mask_shape[3]]
        return tuple(output_shape)


class AddCoords2D(Layer):
    """Add coords to a tensor as described in CoordConv paper.

    # Arguments
        with_r: Boolean flag if the r coordinate is added. See paper for more details.
    
    # Input shape
        4D tensor with shape: (samples, rows, cols, channels)

    # Output shape
        same as input except channels + 2, channels + 3 if with_r is True
    
    # Example
        x = Conv2D(32, 3, padding='same', activation='relu')(x)
        x = AddCoords2D()(x)
        x = Conv2D(32, 3, padding='same', activation='relu')(x)
    
    # References
        https://arxiv.org/abs/1807.03247
    """
    def __init__(self, with_r=False, **kwargs):
        super(AddCoords2D, self).__init__(**kwargs)
        self.with_r = with_r
        
    def call(self, input_tensor):
        """
        input_tensor: (batch_size, x_dim, y_dim, c)
        """
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
    
    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        output_shape[3] = output_shape[3] + 2
        if self.with_r:
            output_shape[3] = output_shape[3] + 1
        return tuple(output_shape)


def Resize2DBilinear(size):
    return Lambda(lambda x: tf.image.resize_bilinear(x, size, align_corners=True))

def Resize2DNearest(size):
    return Lambda(lambda x: tf.image.resize_nearest_neighbor(x, size, align_corners=True))
