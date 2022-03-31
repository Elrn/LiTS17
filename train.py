#-*- coding: utf-8 -*-

import tensorflow as tf
import utils, models, metrics, callbacks, losses
from Data import LiST17, LiST17_2D
from tensorflow.keras.callbacks import *
import os, re, logging

# import tensorflow.experimental.numpy as tnp
# tnp.experimental_enable_numpy_behavior()
########################################################################################################################
os.putenv('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('PIL.PngImagePlugin').disabled = True
logging.getLogger('h5py._conv').disabled = True

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
########################################################################################################################

# init
base_dir = os.path.dirname(os.path.realpath(__file__)) # getcwd()
log_dir = utils.join_dir([base_dir, 'log'])
dirs = ['plt', 'checkpoint']
paths = [utils.join_dir([log_dir, dir]) for dir in dirs]
[utils.mkdir(path) for path in paths]
plt_dir, ckpt_dir = paths

### log
log = logging.getLogger('root')
file_handler = logging.FileHandler(utils.join_dir([log_dir, utils.get_datetime()+'.txt']))
# fh.setLevel(logging.DEBUG)
log.addHandler(file_handler)

### ckpt
ckpt_file_name = 'EP_{epoch}, L_{loss:.3f}, P_{Precision:.3f}, R_{Recall:.3f}, J_{JSC:.3f}, vP_{val_Precision:.3f}, vR_{val_Recall:.3f}, vJ_{val_JSC:.3f}.hdf5'
# ckpt_file_name = 'EP_{epoch}, L_{loss:.4f}, vL_{val_loss:.4f}.hdf5'
ckpt_file_path = utils.join_dir([ckpt_dir, ckpt_file_name])

### Get Data
dataset, val_dataset = LiST17_2D.build(batch_size=10, validation_split=0.2) # [0]:train [1]:valid or None
test_dataset = LiST17_2D.build_test(10)
num_class, input_shape = LiST17_2D.num_class, LiST17_2D.input_shape

### Build model
input = tf.keras.layers.Input(shape=input_shape)
output = models.AE(num_class)(input)
model = tf.keras.Model(input, output, name=None)

### Compile model
metric_list = [
    metrics.Precision(num_class),
    metrics.Recall(num_class),
    metrics.F_Score(num_class),
    metrics.DSC(num_class),
    metrics.JSC(num_class),
]
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=losses.segCE(),
              metrics=metric_list,
              )

### load weights
filepath_to_load = utils.get_checkpoint(ckpt_file_path)
if (filepath_to_load is not None and utils.checkpoint_exists(filepath_to_load)):
    initial_epoch = int(re.findall(r"EP_(\d+),", filepath_to_load)[0])
    try:
        model.load_weights(filepath_to_load)
        print(f'[Model|ckpt] Saved Check point is restored from "{filepath_to_load}".')
    except (IOError, ValueError) as e:
        raise ValueError(f'Error loading file from {filepath_to_load}. Reason: {e}')
else:
    print(f'[Model|ckpt] Model is trained from scratch.')
    initial_epoch = 0

### Train model
history = model.fit(
    x = dataset,
    epochs=100,
    validation_data=val_dataset,
    initial_epoch=initial_epoch,
    callbacks=[
        ModelCheckpoint(ckpt_file_path, monitor='loss', save_best_only=True, save_weights_only=False, save_freq='epoch'),
        # EarlyStopping(monitor='loss', min_delta=0, patience=5),
        # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, min_delta=0.0001, cooldown=0, min_lr=0),
        callbacks.setLR(0.0001),
        callbacks.monitor(plt_dir, dataset=test_dataset)
    ]
)
file_handler.close()
# date = datetime.datetime.today().strftime('%Y-%m-%d_%Hh%Mm%Ss')
# utils.save_history(history, utils.join_dir([base_dir, 'log', date]))