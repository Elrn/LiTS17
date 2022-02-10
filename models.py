import tensorflow as tf
from tensorflow.keras.layers import *

########################################################################################################################
EPSILON = tf.keras.backend.epsilon()
act = 'relu'

BN_ACT = lambda x : tf.nn.relu(BatchNormalization()(x))
ACT_BN = lambda x : BatchNormalization()(tf.nn.relu(x))
########################################################################################################################
class Attention(tf.keras.layers.Layer):
    def __init__(self, input_dims):
        super(Attention, self).__init__()
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
def encoder(filters, kernel=3, pool=2):
    def conv_bn_act(x):
        x = Conv3D(filters, kernel, padding='same', groups=filters)(x)
        x = Conv3D(filters, kernel, padding='same', groups=filters)(x)
        x = BatchNormalization()(x)
        x = Activation(act)(x)
        return x

    def main(inputs):
        x = Conv3D(filters, kernel, padding='same')(inputs)
        x = Attention(1)(x)
        x = conv_bn_act(x)
        x = conv_bn_act(x)
        x = AveragePooling3D(pool, pool)(x)
        return x
    return main

def decoder(filters, div=4, kernel=3):
    concat_list = []
    div_channel = filters // div
    attn = Attention(div)
    def main(x):
        features = [Conv3D(div_channel, 1)(attn(x, i)) for i in range(div)]
        for feature in features:
            x = Conv3DTranspose(div_channel, kernel, strides=2, padding='same')(feature)
            for j in range(1, 4):
                x += Conv3D(div_channel, kernel, dilation_rate=j, padding='same', groups=div_channel)(x)
            concat_list.append(x)
        x = tf.concat(concat_list, -1)
        x = BN_ACT(x)
        return x
    return main

#
def bottle_neck(filters, div=4, kernel=3):
    concat_list = []
    dilation_rate = [1+2*i for i in range(div)]
    div_channel = filters // div
    attn = Attention(div)
    def main(x):
        features = [Conv3D(div_channel, 1)(attn(x, i)) for i in range(div)]
        for i, feature in enumerate(features):
            x = Conv3D(div_channel, kernel, dilation_rate=dilation_rate[i], padding='same')(feature)
            x = Conv3D(div_channel, kernel, dilation_rate=dilation_rate[i], padding='same', groups=div_channel)(x)
            if concat_list:
                x = tf.add(concat_list[-1], x)
            concat_list.append(x)
        x = tf.concat(concat_list, -1)
        x = BN_ACT(x)
        return x
    return main

def build(n_class, depth=2):
    def main(inputs):
        filters = [64*i for i in range(1, depth+2)]
        x = inputs
        # skip_conn_list = []
        ### Encoder
        for i in range(depth):
            x = encoder(filters[i])(x)
            # skip = skip_conn_fn(filters[i])(skip)
            # skip_conn_list.append(skip)
        ### BottleNeck
        x = bottle_neck(filters[-1])(x)
        ### Decoder
        for i in reversed(range(depth)):
            # x = decoder(filters[i])(x, skip_conn_list[i])
            x = decoder(filters[i])(x)
        ### Affine
        x = Conv3D(n_class, 1)(x)
        output = Softmax(-1)(x)
        return output
    return main