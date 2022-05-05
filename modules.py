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
# Encoder
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
# Decoder
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

def context_fusion_block(filters, kernel=3, **kwargs):
    """
    Attention Guided Global Enhancement and Local Refinement Network for Semantic Segmentation
    :return:
    """
    def semantic_distribution_module(filters, kernel=3):
        """
        :return: [B, H, W, C], [B, H, W, C]
        """
        def main(D, B): # [] / [B, N, C]
            D = convTranspose(filters, kernel, strides=2, padding='same', name=f"convt_{kwargs['i']}")(D)
            M = flat()(conv(B.shape[1], 1, padding='same')(D))
            M = tf.matmul(M, B) # B, I, C

            axis = [-1] + list(D.shape[1:])
            M = tf.reshape(M, axis) # B, H, W, C

            S = conv(filters, kernel, padding='same')(D + M)
            S = BatchNormalization()(S)
            S = tf.nn.relu(S)
            return S, M
        return main

    def local_refinement_module():
        """
        :return: [B, H, W, C]
        """
        flatten = flat()
        def main(E, S, M):
            channel_attention = tf.matmul(flatten(E), flatten(S), transpose_a=True) # B, C, C
            F = tf.matmul(flatten(E), channel_attention)
            F = tf.reshape(F, [-1] + list(E.shape[1:]))
            G = F * M
            return G
        return main

    def main(E, D, B):
        S, skip = semantic_distribution_module(filters)(D, B)
        G = local_refinement_module()(E, S, skip)
        x = conv(filters, kernel, padding='same')(S+G)
        x = BatchNormalization()(x)
        x = tf.nn.relu(x)
        return x
    return main

########################################################################################################################
# bottle-neck
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

def sematic_aggregation_block(filters:list, compression_rate):
    """
    Attention Guided Global Enhancement and Local Refinement Network for Semantic Segmentation
    :return: ([B, H, W, C], [B, N, C]:list)
    """
    flatten = flat()
    def main(inputs):
        X = [flatten(conv(f, 1)(inputs)) for f in filters] # [B, HW, C]
        X_t = [flatten(conv(int(f*compression_rate), 1)(inputs)) for f in filters] # [B, HW, N]

        Bs = [tf.matmul(x_t, x, transpose_a=True) for x_t, x in zip(X_t, X)] # [B, N, C]
        return inputs, Bs
    return main

########################################################################################################################
# skip-connection
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