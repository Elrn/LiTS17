import layers
import tensorflow as tf
from tensorflow.keras.layers import *
import modules

def base(filters, kernel=3, pool=2):
    def conv_bn_act(x):
        x = modules.conv(filters, kernel, padding='same', groups=filters)(x)
        x = modules.conv(filters, kernel, padding='same', groups=filters)(x)
        x = modules.BN_ACT(x)
        return x

    def main(inputs):
        x = modules.conv(filters, kernel, padding='same')(inputs)
        x = modules.BN_ACT(x)
        skip = conv_bn_act(x)
        x = modules.pooling(pool, pool)(skip)
        return x, skip
    return main

def base_2(filters, kernel=3, pool=2):
    SP = layers.sep_bias(2)
    def main(x):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = modules.conv(filters, kernel, padding='same')(x)
        x = tf.concat([SP(x, 0), SP(x, 1)], -1)
        x = BatchNormalization()(x)
        x = Activation('gelu')(x)
        x = modules.tranform(x)
        x = BatchNormalization()(x)
        x_ = Activation('relu')(x)
        x = modules.depthwise(kernel, padding='same')(x_)
        x += BatchNormalization()(x)
        x = modules.depthwise(kernel, padding='same')(x)
        skip = x
        if pool != None:
            x = modules.pooling(pool, pool)(x)
        return x, skip
    return main

def multi_scale(filters, kernel=3, pool=2):
    def main(x):
        x = modules.conv(filters, kernel, padding='same')(x)
        x = modules.BN_ACT(x)
        for i in range(1, 4):
            x += modules.conv(filters, kernel, padding='same', groups=filters,
                            dilation_rate=i)(x)
            x = modules.BN_ACT(x)
        x = layers.SB()(x)
        x = tf.nn.relu(x)
        x = modules.depthwise(kernel, padding='same')(x)
        x = layers.SE()(x)
        x = modules.conv(filters, kernel, padding='same', groups=filters)(x)
        skip = modules.BN_ACT(x)
        x = modules.pooling(pool, pool)(skip)
        return x, skip
    return main