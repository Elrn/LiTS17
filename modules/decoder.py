import layers
import tensorflow as tf
from tensorflow.keras.layers import *
import modules



def base(filters, div=4, kernel=3):
    concat_list = []
    div_channel = filters // div
    attn = layers.sep_bias(div)

    def main(x, skip):
        features = [modules.conv(div_channel, 1)(attn(x, i))
                    for i in range(div)]
        for feature in features:
            x = modules.convTranspose(div_channel, kernel, strides=2,
                                     padding='same')(feature)
            for j in range(1, 4):
                x += modules.conv(div_channel, kernel, dilation_rate=j,
                                 padding='same', groups=div_channel)(x)
            concat_list.append(x)
        concat_list.append(skip)
        x = tf.concat(concat_list, -1)
        x = modules.BN_ACT(x)
        return x

    return main


def multi_scale(filters, div=4, kernel=3):
    concat_list = []
    div_channel = filters // div
    attn = layers.sep_bias(div)

    def main(x, skip):
        features = [modules.conv(div_channel, 1)(attn(x, i))
                    for i in range(div)]
        for feature in features:
            x = modules.convTranspose(div_channel, kernel, strides=2,
                                     padding='same')(feature)
            for j in range(1, 4):
                x += modules.conv(div_channel, kernel, dilation_rate=j,
                                 padding='same', groups=div_channel)(x)
            concat_list.append(x)
        concat_list.append(skip)
        x = tf.concat(concat_list, -1)
        x = modules.BN_ACT(x)
        return x
    return main

def context_fusion_block(filters, kernel=3, **kwargs):
    """
    Attention Guided Global Enhancement and Local Refinement Network for Semantic Segmentation
    :return:
    """

    def semantic_distribution_module(filters, kernel=3):
        """
        :return: [B, H, W, C], [B, H, W, C]
        """

        def main(D, B):  # [] / [B, N, C]
            D = modules.convTranspose(filters, kernel, strides=2,
                                     padding='same', name=f"convt_{kwargs['i']}")(D)
            M = layers.flat()(modules.conv(B.shape[1], 1, padding='same')(D))
            M = tf.matmul(M, B)  # B, I, C

            axis = [-1] + list(D.shape[1:])
            M = tf.reshape(M, axis)  # B, H, W, C

            S = modules.conv(filters, kernel, padding='same')(D + M)
            S = BatchNormalization()(S)
            S = tf.nn.relu(S)
            return S, M

        return main

    def local_refinement_module():
        """
        :return: [B, H, W, C]
        """
        flatten = layers.flat()

        def main(E, S, M):
            channel_attention = tf.matmul(flatten(E), flatten(S), transpose_a=True)  # B, C, C
            F = tf.matmul(flatten(E), channel_attention)
            F = tf.reshape(F, [-1] + list(E.shape[1:]))
            G = F * M
            return G

        return main

    def main(E, D, B):
        S, skip = semantic_distribution_module(filters)(D, B)
        G = local_refinement_module()(E, S, skip)
        x = modules.conv(filters, kernel, padding='same')(S + G)
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)
        return x

    return main