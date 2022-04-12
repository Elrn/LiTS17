import tensorflow as tf
from tensorflow.keras.layers import *
from layers import *

########################################################################################################################
act = 'relu'
D = 2
if D == 2:
    conv = Conv2D
    convTranspose = Conv2DTranspose
    pooling = AveragePooling2D
elif D == 3:
    conv = Conv3D
    convTranspose = Conv3DTranspose
    pooling = AveragePooling3D

BN_ACT = lambda x : tf.nn.relu(BatchNormalization()(x))
ACT_BN = lambda x : BatchNormalization()(tf.nn.relu(x))
########################################################################################################################
def encoder(filters, kernel=3, pool=2):
    def conv_bn_act(x):
        x = conv(filters, kernel, padding='same', groups=filters)(x)
        x = conv(filters, kernel, padding='same', groups=filters)(x)
        x = BatchNormalization()(x)
        x = Activation(act)(x)
        return x

    def main(inputs):
        x = conv(filters, kernel, padding='same')(inputs)
        x = BN_ACT(x)
        # x = sep_bias(1)(x)
        x = conv_bn_act(x)
        skip = conv_bn_act(x)
        x = pooling(pool, pool)(skip)
        return x, skip
    return main

########################################################################################################################
def decoder(filters, div=4, kernel=3):
    concat_list = []
    div_channel = filters // div
    attn = sep_bias(div)
    def main(x, skip):
        features = [conv(div_channel, 1)(attn(x, i)) for i in range(div)]
        for feature in features:
            x = convTranspose(div_channel, kernel, strides=2, padding='same')(feature)
            for j in range(1, 4):
                x += conv(div_channel, kernel, dilation_rate=j, padding='same', groups=div_channel)(x)
            concat_list.append(x)
        concat_list.append(skip)
        x = tf.concat(concat_list, -1)
        x = BN_ACT(x)
        return x
    return main

########################################################################################################################
def bottle_neck(filters, div=4, kernel=3):
    concat_list = []
    dilation = [1+2*i for i in range(div)]
    div_channel = filters // div
    attn = sep_bias(div)
    def main(x):
        features = [conv(div_channel, 1)(attn(x, i)) for i in range(div)]
        for i, feature in enumerate(features):
            x = conv(div_channel, kernel, dilation_rate=dilation[i], padding='same')(feature)
            x = conv(div_channel, kernel, dilation_rate=dilation[i], padding='same', groups=div_channel)(x)
            if concat_list:
                x = tf.add(concat_list[-1], x)
            concat_list.append(x)
        x = tf.concat(concat_list, -1)
        x = BN_ACT(x)
        return x
    return main

########################################################################################################################
def skip_connection(filters, kernel=5):
    def main(x):
        x = conv(filters, kernel, padding='same')(x)
        x = conv(filters, 3, padding='same', groups=filters)(x)
        x = conv(filters, 3, padding='same', groups=filters)(x)
        x = BN_ACT(x)
        x = conv(filters, 3, padding='same', groups=filters)(x)
        x = conv(filters, 3, padding='same', groups=filters)(x)
        x = BN_ACT(x)
        return x
    return main

########################################################################################################################