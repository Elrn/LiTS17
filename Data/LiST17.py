"""
https://competitions.codalab.org/competitions/17094#learn_the_details-evaluation
https://paperswithcode.com/dataset/lits17
#	Username	Score
1	DeepwiseAI	0.8220
2	liver_seg	0.7990
3	mumu_all	0.7910

Per-lesion segmentations metrics for detected lesions are the mean values for
Dice score, as well as Jaccard
volume overlap error (VOE),
relative volume difference (RVD),
average symmetric surface distance (ASSD),
maximum symmetric surface distance (MSSD).
"""

"""
volume: ?, 512, 512
segmentation: ?, 512, 512
"""

import tensorflow as tf
from os.path import basename

import numpy as np
import cv2
import utils

num_class = 3
patch_size = 40
input_shape = [patch_size, patch_size, patch_size, 1]
########################################################################################################################
def parse_fn(vol, seg):
    size = [1]+input_shape

    vol = tf.reshape(vol, [1] + vol.shape)  # add batch
    vol = utils.SWN(vol, 30, [150, 25])
    vol = tf.extract_volume_patches(input=vol, ksizes=size, strides=size, padding='VALID')
    vol = tf.reshape(vol, [-1] + input_shape)

    seg = tf.reshape(seg, [1] + seg.shape + [1]) # add batch and channel
    seg = tf.extract_volume_patches(input=seg, ksizes=size, strides=size, padding='VALID')
    seg = tf.reshape(seg, [-1] + input_shape[:-1])  # except channel for one hot
    seg = tf.cast(seg, 'int32')
    seg = tf.one_hot(seg, num_class, axis=-1)
    return (vol, seg)

def validation_split_fn(dataset, validation_split):
    len_dataset = tf.data.experimental.cardinality(dataset).numpy()
    valid_count = int(len_dataset * validation_split)
    print(f'[Dataset|split] Total: "{len_dataset}", Train: "{len_dataset-valid_count}", Valid: "{valid_count}"')
    return dataset.skip(valid_count), dataset.take(valid_count)

def build(batch_size, validation_split=0.1):
    assert 0 <= validation_split <= 0.5
    file_path = 'C:\\dataset\\LiTS17_160_160_200.npz'
    print(f'[Dataset] load:"{basename(file_path)}", batch size:"{batch_size}", split:"{validation_split}"')
    with np.load(file_path) as data:
        dataset = load((data['vol'], data['seg']), batch_size)
        if validation_split is not None and validation_split is not 0:
            return validation_split_fn(dataset, validation_split)
        else:
            return dataset, None

"""
- BATCH Drop Reminder = False
[!] ERR Raised: Non-OK-status: GpuLaunchKernel ... INTERNAL: unspecified launch failure
[!] metrics 적용 가능

https://stackoverflow.com/questions/59068494/program-crashed-in-the-last-step-in-test-tensorflow-gpu-2-0-0/59971264#59971264
the problem is related to the batch size. I am using nvidia docker 19.12 and data generator. 
The code works well with one gpu and the problem happened only with mirroredstrategy in the model.predict.
When the total number of data can not be divided by the batch_size perfectly, the error happens. 
The 3rd batch will have only one data and brings a problem.
The solution is either throw away the last data, or in my case, add some dummy data to fill up the last batch.
"""
def load(data, batch_size, drop=True):
    return tf.data.Dataset.from_tensor_slices(
        data
    # ).prefetch(
    #     tf.data.experimental.AUTOTUNE
    # ).interleave(
    #     lambda x : tf.data.Dataset(x).map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE),
    #     cycle_length = tf.data.experimental.AUTOTUNE,
    #     num_parallel_calls = tf.data.experimental.AUTOTUNE
    ).repeat(
        count=3
    # ).shuffle(
    #     4,
    #     reshuffle_each_iteration=True
    # ).cache(
    ).map(
        map_func=parse_fn,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).unbatch( # batch > unbatch > batch 시 cardinality = -2 로 설정됨
    ).batch(
        batch_size=batch_size,
        drop_remainder=drop,
    )


########################################################################################################################
def build_test(batch_size):
    file_path = 'C:\\dataset\\LiTS17_160_160_200_test.npz'
    with np.load(file_path) as data:
        dataset = load_test((data['vol'], data['seg']), batch_size)
    return dataset

def load_test(data, batch_size, drop=True):
    return tf.data.Dataset.from_tensor_slices(data
        ).map(map_func=parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
        ).unbatch(  # batch > unbatch > batch 시 cardinality = -2 로 설정됨
        ).batch(batch_size=batch_size, drop_remainder=drop,
        )
########################################################################################################################
