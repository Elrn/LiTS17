__all__ = [
    'encoder',
    'latent',
    'decoder',
    'skip_connection',
]

from tensorflow.keras.layers import *
import layers
import tensorflow as tf

conv = None
convTranspose = None
pooling = None
depthwise = None

def assignment_function_according_to_data_rank(rank):
    global conv
    global convTranspose
    global pooling
    global depthwise

    if rank == 2:
        conv = Conv2D
        convTranspose = Conv2DTranspose
        pooling = AveragePooling2D
        depthwise = DepthwiseConv2D
    elif rank == 3:
        conv = Conv3D
        convTranspose = Conv3DTranspose
        pooling = AveragePooling3D
        depthwise = layers.DepthwiseConv3D
    else:
        raise ValueError(f'D is must 2 or 3, not "{rank}".')

def tranform(x):
    SP = layers.sep_bias(2)
    w = LayerNormalization(axis=[1, 2])(SP(x, 0))
    w = tf.nn.softmax(w)
    b = LayerNormalization(axis=[1, 2])(SP(x, 1))
    b = tf.math.sigmoid(b)
    x += w * x + b
    return x


BN_ACT = lambda x : tf.nn.relu(BatchNormalization()(x))
