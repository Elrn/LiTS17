import tensorflow as tf
import os, logging, re
from absl import app
import flags, utils, train, callbacks
FLAGS = flags.FLAGS

########################################################################################################################
def main(argv):
    utils.tf_init()

    if FLAGS.train:
        train.main(argv)
    else: # inference
        base_dir = os.path.dirname(os.path.realpath(__file__))  # getcwd()
        ckpt_dir = utils.join_dir([base_dir, 'log', 'checkpoint']) if FLAGS.ckpt_dir==None else FLAGS.ckpt_dir
        ckpt = callbacks.load_weights._get_most_recently_modified_file_matching_pattern(ckpt_dir)
        if ckpt != None and callbacks.load_weights._checkpoint_exists(ckpt):
            model = tf.keras.models.load_weights(ckpt, custom_objects=None, compile=True, options=None)
        else:
            raise RuntimeError(f'Model Checkpoint is not fount, accepted "{ckpt}".')
        if FLAGS.predict_step:
            model.predict_step()
        else:
            model.predict()

if __name__ == '__main__':
    app.run(main)


