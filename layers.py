
import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.framework.tensor_shape import TensorShape
from tensorflow.python.ops import nn_ops
from tensorflow.python.keras import initializers, regularizers, constraints
from tensorflow.python.keras.utils import conv_utils


class SparseConv2D(Layer):
    """2D sparse convolution layer for sparse input data.
    
    # References
        [Sparsity Invariant CNNs](https://arxiv.org/abs/1708.06500)
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
        
    def build(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape
        feature_shape = TensorShape(feature_shape).as_list()
        
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
        norm = tf.where(tf.equal(norm,0), tf.zeros_like(norm), tf.reciprocal(norm))
        
        features = tf.multiply(features, norm)
        if self.use_bias:
            features = tf.add(features, self.bias)
            
        mask = nn_ops.pool(mask, self.kernel_size, 'MAX', self.padding.upper(), self.dilation_rate, self.strides)
        
        return [features, mask]
    
    def compute_output_shape(self, input_shape):
        if type(input_shape) is list:
            feature_shape = input_shape[0]
        else:
            feature_shape = input_shape
        feature_shape = TensorShape(feature_shape).as_list()
        
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
        
        return [TensorShape(feature_shape), TensorShape(mask_shape)]


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
        input_shape = TensorShape(input_shape).as_list()
        output_shape = [dim // ratio[idx] if dim is not None else None for idx, dim in enumerate(input_shape)]
        output_shape = tuple(output_shape)
        return [TensorShape(output_shape), TensorShape(output_shape)]

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
            self.output_shape1 = output_shape
            
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
        mask_shape = TensorShape(input_shape[1]).as_list()
        output_shape = [mask_shape[0], mask_shape[1] * self.size[0], mask_shape[2] * self.size[1], mask_shape[3]]
        return TensorShape(output_shape)

