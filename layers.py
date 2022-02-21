import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import conv_utils

########################################################################################################################
EPSILON = tf.keras.backend.epsilon()
act = 'relu'

BN_ACT = lambda x : tf.nn.relu(BatchNormalization()(x))
ACT_BN = lambda x : BatchNormalization()(tf.nn.relu(x))

########################################################################################################################
class LAP_2D(pooling.Pooling2D):
    """ https://arxiv.org/abs/2201.11808
    LAP: An Attention-Based Module for Faithful Interpretation and Knowledge Injection in Convolutional Neural Networks
    """
    def __init__(self, pool_size=2, strides=2, padding='VALID', data_format=None, name=None, **kwargs):
        super(LAP_2D, self).__init__(
            self.pool_function, pool_size=pool_size, strides=strides, padding=padding, name=name,
            data_format=data_format, **kwargs
        )
    def build(self, input_shape):
        self.n_ch = input_shape[-1]
        self.alpha = self.add_weight("alpha", shape=[1], initializer='HeNormal')

    def pool_function(self, input, ksize, strides, padding, data_format=None):
        if data_format == 'channels_first':
            input = tf.transpose(input, [0, 2, 3, 1])
        patches = tf.image.extract_patches(input, ksize, strides, [1, 1, 1, 1], padding)
        patches = [tf.strided_slice(patches, [0, 0, 0, i], patches.shape, [1, 1, 1, self.n_ch])
                   for i in range(self.n_ch)]
        max = tf.reduce_max(patches, -1, keepdims=True)
        prob = tf.math.exp(-(self.alpha ** 2) * (max - patches) ** 2)
        prob = (prob * patches + EPSILON)
        mean = tf.reduce_mean(prob, -1, keepdims=True)
        mean = tf.concat(tf.split(mean, self.n_ch, 0), -1)
        mean = tf.squeeze(mean, 0)
        return mean

########################################################################################################################
class SaBN(Layer):
    def __init__(self, n_class, axis=-1):
        super(SaBN, self).__init__()
        self.n_class = n_class
        self.BN = BatchNormalization(axis=axis)
        self.axis = [axis] if isinstance(axis, int) else axis

    def build(self, input_shape):
        x_shape, _ = input_shape # except y(label)
        param_shape = self.get_param_shape(input_shape)

        self.scale = self.add_weight("scale", shape=param_shape, initializer='ones')
        self.offset = self.add_weight("offset", shape=param_shape, initializer='zeros')

    def get_param_shape(self, input_shape):
        ndims = len(input_shape)
        # negative parameter to positive parameter
        axis = [ndims + ax if ax < 0 else ax for ax in self.axis]
        axis_to_dim = {x: input_shape[x] for x in axis}
        param_shape = [axis_to_dim[i] if i in axis_to_dim else 1 for i in range(ndims)]
        return param_shape

    def call(self, inputs, training=None, **kargs):
        if training == False:
            return self.BN(inputs, training=training)
        assert len(inputs) == 2
        x, y = inputs
        out = self.BN(x, training=training)
        return self.scale * out + self.offset

    def get_config(self):
        config = super().get_config()
        config.update({
            "scale": self.scale,
            "offset": self.offset,
        })
        return config

########################################################################################################################
class sep_bias(Layer):
    def __init__(self, input_dims):
        super(sep_bias, self).__init__()
        assert input_dims > 0
        self.input_dims = input_dims

    def build(self, input_shape):
        self.scale = Embedding(self.input_dims, input_shape[-1], embeddings_initializer='ones')
        self.offset = Embedding(self.input_dims, input_shape[-1], embeddings_initializer='zeros')

    def call(self, inputs, label=0, training=None):
        assert self.input_dims >= label
        return self.scale(label) * inputs + self.offset(label)

    def get_config(self):
        config = super().get_config()
        config.update({
            "scale": self.scale,
            "offset": self.offset,
        })
        return config

########################################################################################################################