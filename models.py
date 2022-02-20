import tensorflow as tf
from tensorflow.keras.layers import *
import layers, modules

########################################################################################################################
def AE(n_class, base_filters=64, depth=2):
    def main(inputs):
        filters = [base_filters*i for i in range(1, depth+2)]
        x = inputs
        # skip_conn_list = []
        ### Encoder
        for i in range(depth):
            x = modules.encoder(filters[i])(x)
            # skip = skip_conn_fn(filters[i])(skip)
            # skip_conn_list.append(skip)
        ### BottleNeck
        x = modules.bottle_neck(filters[-1])(x)
        ### Decoder
        for i in reversed(range(depth)):
            # x = decoder(filters[i])(x, skip_conn_list[i])
            x = modules.decoder(filters[i])(x)
        ### Affine
        x = Conv3D(n_class, 1)(x)
        output = Softmax(-1)(x)
        return output
    return main

########################################################################################################################