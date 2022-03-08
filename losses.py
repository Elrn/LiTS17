import logging

from tensorflow.python.keras.losses import *
from keras.utils import losses_utils
import tensorflow as tf
from tensorflow import reduce_mean as E

import numpy as np

########################################################################################################################
EPSILON = tf.keras.backend.epsilon()
# from keras.backend import categorical_crossentropy

########################################################################################################################
def EL(y_true, y_pred, frequences=None, gamma=0.3): # B, H, W, C
    """
    3D Segmentation with Exponential Logarithmic Loss for Highly Unbalanced Object Sizes
        https://arxiv.org/abs/1809.00076


    :param frequences: Imbalancing에 대한 class별 weight 값
    """
    logging.warning(f"[Loss] EL loss's gamma recommended '0 ~ 1', but '{gamma}'.")
    assert len(frequences) == y_true.shape[-1]
    axis = [i for i in range(len(y_true.shape) - 1)] # except Batch, Channel

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

    def get_DICE(gamma=0.3):
        y_ = tf.one_hot(tf.argmax(y_pred, -1), y_pred.shape[-1])
        delta = tf.where(tf.math.equal(y_true, y_), 1, 0)
        numerator = 2 * tf.reduce_sum(delta * y_pred, axis) + EPSILON
        denominator = tf.reduce_sum(delta + y_pred, axis) + EPSILON

        DICE = -tf.math.log(numerator / denominator)
        DICE = DICE ** gamma
        return E(DICE, -1)

    def get_CE(frequences, gamma=0.3):
        CE = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis)
        CE = (frequences * CE) ** gamma
        return E(CE)

    if frequences == None:
        frequences = get_frequences_by_class(y_true)

    DICE = get_DICE()
    CE = get_CE(frequences)
    return DICE + CE

########################################################################################################################
def focal(y_true, y_pred, gamma=0.3):
    """
    Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002
    """
    axis = [i for i in range(len(y_true.shape)-1)]
    loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis)
    loss *= (1 - y_true * y_pred) ** gamma
    return loss

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
