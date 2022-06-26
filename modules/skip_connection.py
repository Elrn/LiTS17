import modules
from tensorflow.keras.layers import *
import tensorflow as tf
import layers

def base(filters, kernel=5):
    def main(x):
        x = modules.conv(filters, kernel, padding='same')(x)
        x = modules.conv(filters, 3, padding='same', groups=filters)(x)
        x = modules.conv(filters, 3, padding='same', groups=filters)(x)
        x = modules.BN_ACT(x)
        x = modules.conv(filters, 3, padding='same', groups=filters)(x)
        x = modules.conv(filters, 3, padding='same', groups=filters)(x)
        x = modules.BN_ACT(x)
        return x
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
        x = modules.transform(x)
        x = BatchNormalization()(x)
        x_ = Activation('relu')(x)
        x = modules.depthwise(kernel, padding='same')(x_)
        x = BatchNormalization()(x)
        x = modules.depthwise(kernel, padding='same')(x) + x_
        return x
    return main