import logging

from tensorflow.python.keras.losses import *
from keras.utils import losses_utils
import tensorflow as tf
from tensorflow import reduce_mean as E

import numpy as np
from scipy import ndimage

########################################################################################################################
EPSILON = tf.keras.backend.epsilon()
# from keras.backend import categorical_crossentropy

########################################################################################################################
def Cross_Entropy(y_true, y_pred):
    def get_mask(y_pred):
        """ return mask(0, 1) corresponding to prediction """
        n_ch = y_pred.shape[-1]
        y_pred = tf.math.argmax(y_pred, -1)
        y_pred = tf.one_hot(y_pred, n_ch)
        return y_pred

    prediction_mask = get_mask(y_pred)
    condition = (y_true - prediction_mask) < 0  # FP
    y_pred = tf.where(condition, 1 - y_pred, y_pred)
    loss = prediction_mask * -tf.math.log(y_pred)  # only part of prediction
    return loss # B, H, W, C

########################################################################################################################
def dist_CE(distance_rate=0.04):
    """
    patch의 경우 상대적 distance 추가 필요?
    """
    def get_distance_weight(y_true):
        y_true = ndimage.distance_transform_edt(y_true < 1, distance_rate) + 1
        return y_true

    def frequences_by_class(y_true):
        def weight_fn(x):
            # x = (-tf.math.log(x)) ** 0.5 + 0.2
            # x = -tf.math.sin(x-1)+0.5
            x = -x + 1.4
            return x
        total = np.prod(y_true.shape[:-1])
        axis = [i for i in range(len(y_true.shape) - 1)]
        amount_by_class = tf.reduce_sum(y_true, axis, keepdims=True)
        # add EPSILON to prevent inf value
        weight = weight_fn((amount_by_class + EPSILON) / (total + EPSILON))  # B, 1, 1, nC
        weight = tf.reduce_max(weight * y_true, -1, keepdims=True)
        return weight

    def main(y_true, y_pred):
        distance_weight_map = get_distance_weight(y_true)
        frequency_weight_map = frequences_by_class(y_true)
        CE = Cross_Entropy(y_true, y_pred)

        loss = frequency_weight_map * distance_weight_map * CE
        return loss
    return main

########################################################################################################################
def EL(y_true, y_pred, frequences=None, gamma=0.3): # B, H, W, C
    """
    3D Segmentation with Exponential Logarithmic Loss for Highly Unbalanced Object Sizes
        https://arxiv.org/abs/1809.00076

    :param frequences: Imbalancing에 대한 class별 weight 값
    """
    def get_frequences_by_class(x):
        """
        patches 혹은 slices로 학습을 진행할 때
        병변 부위가 없는 batch의 경우 imbalance label overfitting 우려 존재
        따라서 수동 기입을 권장
        """
        weight_fn = lambda x: (-np.log(x))**0.5

        total = np.prod(x.shape[:-1])
        amount_by_class = tf.reduce_sum(x, [i for i in range(len(x.shape)-1)])

        return weight_fn(amount_by_class / total)

    if frequences == None:
        frequences = get_frequences_by_class(y_true)
        assert len(frequences) == y_true.shape[-1]
    logging.warning(f"[Loss] EL loss's gamma recommended '0 ~ 1', but '{gamma}'.")
    axis = [i for i in range(len(y_true.shape) - 1)] # except Batch, Channel

    def get_DICE(gamma=0.3):
        y_ = tf.one_hot(tf.argmax(y_pred, -1), y_pred.shape[-1])
        delta = tf.where(tf.math.equal(y_true, y_), 1, 0)
        numerator = 2 * tf.reduce_sum(delta * y_pred, axis) + EPSILON
        denominator = tf.reduce_sum(delta + y_pred, axis) + EPSILON

        DICE = -tf.math.log(numerator / denominator)
        DICE = DICE ** gamma
        return E(DICE, -1)

    DICE = get_DICE()
    CE = Cross_Entropy(y_true, y_pred)
    CE = (frequences * CE) ** gamma
    return DICE + CE

########################################################################################################################
def focal(y_true, y_pred, gamma=0.3):
    """
    Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002
    """
    CE = Cross_Entropy(y_true, y_pred)
    return CE * (1 - y_true * y_pred) ** gamma

########################################################################################################################
# class custom_loss_template(LossFunctionWrapper):
#     def __init__(self,
#                  from_logits=False,
#                  label_smoothing=0.,
#                  axis=-1,
#                  reduction=losses_utils.ReductionV2.AUTO,
#                  name='custom_loss_template'
#                  ):
#         super().__init__(
#             self.SR,
#             name=name,
#             reduction=reduction,
#             from_logits=from_logits,
#             label_smoothing=label_smoothing,
#             axis=axis
#         )
#
#     def SR(self, y_true, y_pred, from_logits=False, label_smoothing=0., axis=-1, tau=2):
#         y_pred = tf.convert_to_tensor(y_pred)
#         y_true = tf.cast(y_true, y_pred.dtype)
#         label_smoothing = tf.convert_to_tensor(label_smoothing, dtype=y_pred.dtype)
#
#         y_true = tf.nn.softmax(y_true) ** (1 / tau)
#         y_pred = tf.nn.softmax(y_pred) ** (1 / tau)
#
#         def _smooth_labels():
#             num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)
#             return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)
#
#         y_true = tf.__internal__.smart_cond.smart_cond(label_smoothing, _smooth_labels,
#                                                        lambda: y_true)
#
#         return tau**2 * backend.categorical_crossentropy(
#             y_true, y_pred, from_logits=from_logits, axis=axis)

########################################################################################################################
