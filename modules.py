import tensorflow as tf
from tensorflow.keras.layers import *
from layers import *

########################################################################################################################
act = 'relu'

BN_ACT = lambda x : tf.nn.relu(BatchNormalization()(x))
ACT_BN = lambda x : BatchNormalization()(tf.nn.relu(x))
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
        x = sep_bias(1)(x)
        x = conv_bn_act(x)
        x = conv_bn_act(x)
        x = AveragePooling3D(pool, pool)(x)
        return x
    return main

########################################################################################################################
def decoder(filters, div=4, kernel=3):
    concat_list = []
    div_channel = filters // div
    attn = sep_bias(div)
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

########################################################################################################################
def bottle_neck(filters, div=4, kernel=3):
    concat_list = []
    dilation_rate = [1+2*i for i in range(div)]
    div_channel = filters // div
    attn = sep_bias(div)
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

########################################################################################################################
def skip_connection(filters):
    def main(x):
        return x
    return main

########################################################################################################################