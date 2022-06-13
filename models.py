import modules.__init__
import tensorflow as tf
from tensorflow.keras.layers import *
import layers
from modules import *

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
def base(n_class, base_filters=64, depth=3):
    def main(inputs):
        filters = [base_filters*i for i in range(1, depth+2)]
        x = inputs
        skip_conn_list = []
        ### Encoder
        for i in range(depth):
            x, skip = encoder.multi_scale(filters[i])(x)
            skip = skip_connection.base(filters[i])(skip)
            skip_conn_list.append(skip)
        ### BottleNeck
        x = latent.base(filters[-1])(x)
        ### Decoder
        for i in reversed(range(depth)):
            x = decoder.base(filters[i])(x, skip_conn_list[i])
        x = modules.conv(n_class, 1)(x)
        output = Softmax(-1)(x)
        return output
    return main

########################################################################################################################
# def tmp(n_class, base_filters=64, depth=3):
#     def main(inputs):
#         filters = [base_filters*i for i in range(1, depth+2)]
#         x = inputs
#         skip_conn_list = []
#         ### Encoder
#         for i in range(depth):
#             x, skip = encoder.multi_scale(filters[i])(x)
#             skip = skip_connection.base(filters[i])(skip)
#             skip_conn_list.append(skip)
#         ### BottleNeck
#         x = latent.base(filters[-1])(x)
#         ### Decoder
#         for i in reversed(range(depth)):
#             x = decoder.base(filters[i])(x, skip_conn_list[i])
#         x = modules.conv(n_class, 1)(x)
#         output = Softmax(-1)(x)
#         return output
#     return main

########################################################################################################################

def AGLN(n_class, base_filters=64, depth=3, compression_rate=0.6):
    """
    Attention Guided Global Enhancement and Local Refinement Network for Semantic Segmentation

    self-attention을 사용한 U-Net model
    """
    def main(x):
        filters = [base_filters * i for i in range(1, depth + 2)]
        skips = []
        ### Encoder
        for i in range(depth):
            x, skip = encoder.base(filters[i])(x)
            skips.append(skip_connection(filters[i])(skip))
        ### BottleNeck
        x = skip_connection.base(filters[-1])(x)
        D, Bs = latent.sematic_aggregation_block(filters, compression_rate)(x)
        ### Decoder
        for i in reversed(range(depth)):
            D = decoder.context_fusion_block(filters[i], i=i)(skips[i], D, Bs[i])
        x = modules.conv(n_class, 1)(D)
        output = Softmax(-1)(x)
        return output
    return main