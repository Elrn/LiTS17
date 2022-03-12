import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.python.keras.layers import pooling
from tensorflow.keras.constraints import *
from tensorflow.keras.initializers import *

from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops

import functools
import numpy as np

########################################################################################################################
EPSILON = tf.keras.backend.epsilon()

########################################################################################################################
class LAP_2D(pooling.Pooling2D):
    """
    LAP: An Attention-Based Module for Faithful Interpretation and Knowledge Injection in Convolutional Neural Networks
        https://arxiv.org/abs/2201.11808
    """
    def __init__(self, pool_size=2, strides=2, padding='VALID', data_format=None, name=None, **kwargs):
        super(LAP_2D, self).__init__(
            self.pool_function, pool_size=pool_size, strides=strides, padding=padding, name=name,
            data_format=data_format, **kwargs
        )
    def build(self, input_shape):
        self.n_ch = input_shape[-1]
        self.alpha = self.add_weight("alpha", shape=[1], initializer=Constant(0.1), constraint=NonNeg())

    def pool_function(self, input, ksize, strides, padding, data_format=None):
        if data_format == 'channels_first':
            input = tf.transpose(input, [0, 2, 3, 1])
        patches = tf.image.extract_patches(input, ksize, strides, [1, 1, 1, 1], padding)
        max = tf.nn.max_pool(input, ksize, strides, padding)
        softmax = tf.math.exp(-(self.alpha ** 2) * (tf.repeat(max, tf.reduce_prod(ksize), -1) - patches) ** 2)
        logit = (softmax * patches + EPSILON)
        # channel에 따라 patch가 섞이므로 분리 후 재 결합
        logit = tf.stack(tf.split(logit, logit.shape[-1] // self.n_ch, -1), -1)
        return tf.reduce_mean(logit, -1)

########################################################################################################################
class AdaPool(pooling.Pooling2D):
    """
    AdaPool: Exponential Adaptive Pooling for Information-Retaining Downsampling
        https://arxiv.org/abs/2111.00772
    """
    def __init__(self, pool_size=2, strides=2, padding='VALID', data_format=None, name=None, **kwargs):
        super(AdaPool, self).__init__(
            self.pool_function, pool_size=pool_size, strides=strides, padding=padding, name=name,
            data_format=data_format, **kwargs
        )
    def build(self, input_shape):
        self.n_ch = input_shape[-1]
        self.beta = self.add_weight(
            "beta", shape=[1], initializer=Constant(0.5), constraint=MinMaxNorm(min_value=0.0, max_value=1.0)
        )
    def eMPool(self, x, axis=-1):
        x *= tf.nn.softmax(x, axis)
        return tf.reduce_sum(x, axis)

    def eDSCWPool(self, x, axis=-1):
        DSC = lambda x, x_: tf.math.abs(2 * (x * x_)) / (x ** 2 + x_ ** 2 + EPSILON)
        x_ = tf.reduce_mean(x, axis, keepdims=True)
        dsc = tf.math.exp(DSC(x, x_))
        output = dsc * x / tf.reduce_sum(dsc, axis, keepdims=True)
        return tf.reduce_sum(output, axis)

    def pool_function(self, input, ksize, strides, padding, data_format=None):
        if data_format == 'channels_first':
            input = tf.transpose(input, [0, 2, 3, 1])
        patches = tf.image.extract_patches(input, ksize, strides, [1, 1, 1, 1], padding)
        patches = tf.stack(tf.split(patches, patches.shape[-1] // self.n_ch, -1), -1)
        return self.eMPool(patches) * self.beta + self.eDSCWPool(patches) * (1 - self.beta)

########################################################################################################################
class SaBN(Layer):
    """
    Sandwich Batch Normalization: A Drop-In Replacement for Feature Distribution Heterogeneity
        https://arxiv.org/abs/2102.11382
    """
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
class LI(tf.python.keras.layers.convolutional.Conv2D):
    """
    Dilated Convolutions with Lateral Inhibitions for Semantic Image Segmentation
        https://arxiv.org/abs/2006.03708
    """
    def __init__(self,
                 filters,
                 kernel_size=3,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 groups=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer='zeros',
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 intensity=0.2,
                 **kwargs):
        super(LI, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs
        )
        self.intensity = intensity

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if len(input_shape) != 4:
            raise ValueError('Inputs should have rank 4. Received input '
                             'shape: ' + str(input_shape))
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                'The number of input channels must be evenly divisible by the number '
                'of groups. Received groups={}, but the input has {} channels '
                '(full input shape is {}).'.format(self.groups, input_channel,
                                                   input_shape))
        kernel_shape = self.kernel_size + (input_channel // self.groups, self.filters)

        self.weight = self.add_weight("weight", shape=input_channel, initializer='HeNormal')

        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        self.built = True

    def distance_measure(self, size=3):
        """ Euclidean distance """
        x = np.mgrid[:size, :size]
        x = (x - size // 2) * 1.0

        # x = (x ** 2).sum(0) ** 0.5
        x = tf.norm(x, axis=0)
        return x

    def call(self, inputs, label=0, training=None):
        input_shape = inputs.shape

        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

        outputs = self._convolution_op(inputs, self.kernel)

        if self.use_bias:
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                # nn.bias_add does not accept a 1D input tensor.
                bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                # Handle multiple batch dimensions.
                if output_rank is not None and output_rank > 2 + self.rank:
                    def _apply_fn(o):
                        return nn.bias_add(o, self.bias, data_format=self._tf_data_format)
                    outputs = conv_utils.squeeze_batch_dims(
                        outputs, _apply_fn, inner_rank=self.rank + 1)
                else:
                    outputs = nn.bias_add(
                        outputs, self.bias, data_format=self._tf_data_format)

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(input_shape)
            outputs.set_shape(out_shape)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({
            "weight": self.weight,
        })
        return config

########################################################################################################################
class channel_attention_base(Layer):
    def __init__(self, squeeze_rate=0.6):
        super(channel_attention_base, self).__init__()
        self.squeeze_rate = squeeze_rate

    def build(self, input_shape):
        self.n_channel = input_shape[self._get_channel_axis()]
        rank = len(input_shape)
        self.GAP = GlobalAveragePooling2D(keepdims=True) if rank == 4 \
            else GlobalAveragePooling3D(keepdims=True)

class SE(channel_attention_base):
    """
    Squeeze-and-Excitation Networks
        https://arxiv.org/abs/1709.01507
    """
    def __init__(self, squeeze_rate=0.6):
        super(SE, self).__init__()
        self.squeeze_rate = squeeze_rate

    def call(self, inputs, training=None, **kargs):
        x = self.GAP(inputs)

        x = Dense(int(self.n_channel * self.squeeze_rate))(x)
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = Dense(self.n_channel)(x)
        x = BatchNormalization()(x)
        x = tf.math.sigmoid(x)
        return inputs * x

class SB(channel_attention_base):
    """
    Shift-and-Balance Attention
        https://arxiv.org/abs/2103.13080
    """
    def __init__(self, squeeze_rate=0.6):
        super(SB, self).__init__()
        self.squeeze_rate = squeeze_rate

    def call(self, inputs, training=None, **kargs):
        x = self.GAP(inputs)

        x = Dense(int(self.n_channel * self.squeeze_rate))(x)
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)

        x = Dense(self.n_channel)(x)
        x = BatchNormalization()(x)
        x = tf.math.tanh(x)
        return inputs + x


########################################################################################################################
########################################################################################################################
class Conv(Layer):
    def __init__(self,
                 rank,filters,kernel_size,strides=1,padding='valid',data_format=None,dilation_rate=1,groups=1,               activation=None,use_bias=True,kernel_initializer='glorot_uniform',               bias_initializer='zeros',
                 kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,
                 bias_constraint=None,trainable=True,name=None,conv_op=None,**kwargs):
        super(Conv, self).__init__(trainable=trainable, name=name,
                                   activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        self.rank = rank
        if isinstance(filters, float):
            filters = int(filters)
        if filters is not None and filters < 0:
            raise ValueError(f'Received a negative value for `filters`.'
                             f'Was expecting a positive value, got {filters}.')
        self.filters = filters
        self.groups = groups or 1
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')

        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=self.rank + 2)

        self._validate_init()
        self._is_causal = self.padding == 'causal'
        self._channels_first = self.data_format == 'channels_first'
        self._tf_data_format = conv_utils.convert_data_format(
            self.data_format, self.rank + 2)

    def _validate_init(self):
        if self.filters is not None and self.filters % self.groups != 0:
            raise ValueError(
                'The number of filters must be evenly divisible by the number of '
                'groups. Received: groups={}, filters={}'.format(
                    self.groups, self.filters))

        if not all(self.kernel_size):
            raise ValueError('The argument `kernel_size` cannot contain 0(s). '
                             'Received: %s' % (self.kernel_size,))

        if not all(self.strides):
            raise ValueError('The argument `strides` cannot contains 0(s). '
                             'Received: %s' % (self.strides,))

        if (self.padding == 'causal' and not isinstance(self,
                                                        (Conv1D, SeparableConv1D))):
            raise ValueError('Causal padding is only supported for `Conv1D`'
                             'and `SeparableConv1D`.')

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        input_channel = self._get_input_channel(input_shape)
        if input_channel % self.groups != 0:
            raise ValueError(
                'The number of input channels must be evenly divisible by the number '
                'of groups. Received groups={}, but the input has {} channels '
                '(full input shape is {}).'.format(self.groups, input_channel,
                                                   input_shape))
        kernel_shape = self.kernel_size + (input_channel // self.groups,
                                           self.filters)

        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype)
        else:
            self.bias = None
        channel_axis = self._get_channel_axis()
        self.input_spec = InputSpec(min_ndim=self.rank + 2,
                                    axes={channel_axis: input_channel})

        # Convert Keras formats to TF native formats.
        if self.padding == 'causal':
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance(self.padding, str):
            tf_padding = self.padding.upper()
        else:
            tf_padding = self.padding
        tf_dilations = list(self.dilation_rate)
        tf_strides = list(self.strides)

        tf_op_name = self.__class__.__name__
        if tf_op_name == 'Conv1D':
            tf_op_name = 'conv1d'  # Backwards compat.

        self._convolution_op = functools.partial(
            nn_ops.convolution_v2,
            strides=tf_strides,
            padding=tf_padding,
            dilations=tf_dilations,
            data_format=self._tf_data_format,
            name=tf_op_name)
        self.built = True

    def call(self, inputs):
        input_shape = inputs.shape

        if self._is_causal:  # Apply causal padding to inputs for Conv1D.
            inputs = array_ops.pad(inputs, self._compute_causal_padding(inputs))

        outputs = self._convolution_op(inputs, self.kernel)

        if self.use_bias:
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first:
                # nn.bias_add does not accept a 1D input tensor.
                bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                outputs += bias
            else:
                # Handle multiple batch dimensions.
                if output_rank is not None and output_rank > 2 + self.rank:

                    def _apply_fn(o):
                        return nn.bias_add(o, self.bias, data_format=self._tf_data_format)

                    outputs = conv_utils.squeeze_batch_dims(
                        outputs, _apply_fn, inner_rank=self.rank + 1)
                else:
                    outputs = nn.bias_add(
                        outputs, self.bias, data_format=self._tf_data_format)

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(input_shape)
            outputs.set_shape(out_shape)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def _spatial_output_shape(self, spatial_input_shape):
        return [
            conv_utils.conv_output_length(
                length,
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i])
            for i, length in enumerate(spatial_input_shape)
        ]

    def compute_output_shape(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        batch_rank = len(input_shape) - self.rank - 1
        if self.data_format == 'channels_last':
            return tensor_shape.TensorShape(
                input_shape[:batch_rank]
                + self._spatial_output_shape(input_shape[batch_rank:-1])
                + [self.filters])
        else:
            return tensor_shape.TensorShape(
                input_shape[:batch_rank] + [self.filters] +
                self._spatial_output_shape(input_shape[batch_rank + 1:]))

    def _recreate_conv_op(self, inputs):  # pylint: disable=unused-argument
        return False

    def get_config(self):
        config = {
            'filters':
                self.filters,
            'kernel_size':
                self.kernel_size,
            'strides':
                self.strides,
            'padding':
                self.padding,
            'data_format':
                self.data_format,
            'dilation_rate':
                self.dilation_rate,
            'groups':
                self.groups,
            'activation':
                activations.serialize(self.activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint)
        }
        base_config = super(Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _compute_causal_padding(self, inputs):
        """Calculates padding for 'causal' option for 1-d conv layers."""
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if getattr(inputs.shape, 'ndims', None) is None:
            batch_rank = 1
        else:
            batch_rank = len(inputs.shape) - 2
        if self.data_format == 'channels_last':
            causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
        return causal_padding

    def _get_channel_axis(self):
        if self.data_format == 'channels_first':
            return -1 - self.rank
        else:
            return -1

    def _get_input_channel(self, input_shape):
        channel_axis = self._get_channel_axis()
        if input_shape.dims[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        return int(input_shape[channel_axis])

    def _get_padding_op(self):
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding
        if not isinstance(op_padding, (list, tuple)):
            op_padding = op_padding.upper()
        return op_padding
