import layers
import tensorflow as tf
import modules
from tensorflow.keras.layers import *

def base(filters, div=4, kernel=3):
    concat_list = []
    dilation = [1+2*i for i in range(div)]
    div_channel = filters // div
    attn = layers.sep_bias(div)
    def main(x):
        features = [modules.conv(div_channel, 1)(attn(x, i)) for i in range(div)]
        for i, feature in enumerate(features):
            x = modules.conv(div_channel, kernel, dilation_rate=dilation[i],
                            padding='same')(feature)
            x = modules.conv(div_channel, kernel, dilation_rate=dilation[i],
                            padding='same', groups=div_channel)(x)
            if concat_list:
                x = tf.add(concat_list[-1], x)
            concat_list.append(x)
        x = tf.concat(concat_list, -1)
        x = modules.BN_ACT(x)
        return x
    return main

def base_2(filters, div=4, kernel=3):
    concat_list = []
    dilation = [1+2*i for i in range(div)]
    div_channel = filters // div
    SP = layers.sep_bias(div)
    def main(x):
        x = tf.nn.relu(BatchNormalization()(x))
        features = [Dense(div_channel)(SP(x, i)) for i in range(div)]
        for i, feature in enumerate(features):
            x = modules.conv(div_channel, kernel, dilation_rate=dilation[i],
                            padding='same')(feature)
            x = tf.nn.relu(BatchNormalization()(x))
            x = modules.conv(div_channel, kernel, dilation_rate=dilation[i],
                            padding='same', groups=div_channel)(x)
            x = Dense(div_channel)(layers.sep_bias(1)(x))
            if concat_list:
                x = tf.add(concat_list[-1], x)
            concat_list.append(x)
        x = tf.concat(concat_list, -1)
        return x
    return main

def sematic_aggregation_block(filters:list, compression_rate):
    """
    Attention Guided Global Enhancement and Local Refinement Network for Semantic Segmentation
    :return: ([B, H, W, C], [B, N, C]:list)
    """
    flatten = layers.flat()
    def main(inputs):
        X = [flatten(modules.conv(f, 1)(inputs))
             for f in filters] # [B, HW, C]
        X_t = [flatten(modules.conv(int(f*compression_rate), 1)(inputs))
               for f in filters] # [B, HW, N]

        Bs = [tf.matmul(x_t, x, transpose_a=True)
              for x_t, x in zip(X_t, X)] # [B, N, C]
        return inputs, Bs
    return main
