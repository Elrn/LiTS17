import tensorflow as tf
from tensorflow.keras.layers import *
import layers, modules

########################################################################################################################


########################################################################################################################
def SR(n_class, base_filters=64, depth=2):
    """
    Self-Regulation for Semantic Segmentation
        https://arxiv.org/abs/2108.09702
    """
    exits = []
    def main(inputs):
        return
    return main

########################################################################################################################
def AE(n_class, base_filters=64, depth=3):
    def main(inputs):
        filters = [base_filters*i for i in range(1, depth+2)]
        x = inputs
        skip_conn_list = []
        ### Encoder
        for i in range(depth):
            x, skip = modules.encoder(filters[i])(x)
            skip = modules.skip_connection(filters[i])(skip)
            skip_conn_list.append(skip)
        ### BottleNeck
        x = modules.bottle_neck(filters[-1])(x)
        ### Decoder
        for i in reversed(range(depth)):
            x = modules.decoder(filters[i])(x, skip_conn_list[i])
        x = modules.conv(n_class, 1)(x)
        output = Softmax(-1)(x)
        return output
    return main

########################################################################################################################

def AGLN(n_class, base_filters=32, depth=3, compression_rate=0.6):
    """
    Attention Guided Global Enhancement and Local Refinement Network for Semantic Segmentation
    :return:
    """
    def main(x):
        filters = [base_filters * i for i in range(1, depth + 2)]
        skips = []
        ### Encoder
        for i in range(depth):
            x, skip = modules.encoder(filters[i])(x)
            skips.append(modules.skip_connection(filters[i])(skip))
        ### BottleNeck
        x = modules.bottle_neck(filters[-1])(x)
        D, Bs = modules.sematic_aggregation_block(filters, compression_rate)(x)
        ### Decoder
        for i in reversed(range(depth)):
            D = modules.context_fusion_block(filters[i], i=i)(skips[i], D, Bs[i])
        x = modules.conv(n_class, 1)(D)
        output = Softmax(-1)(x)
        return output
    return main