# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR,'utils_xyz'))

from absl import app as absl_app
from absl import flags
import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.flags import core as flags_core
import resnet_model
import resnet_run_loop
import os, glob, sys

BASE_DIR = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from dataset_utils import parse_pl_record

_DATA_PARAS = None

_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = 40

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 10000

DATASET_NAME = 'MODELNET40'

###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  assert os.path.exists(data_dir), ('not exsit: %s'%(data_dir))
  if is_training:
    return glob.glob(os.path.join(data_dir, '*.tfrecord'))
  else:
    return glob.glob(os.path.join(data_dir, '*.tfrecord'))


def input_fn(is_training, data_dir, batch_size, data_paras=None, num_epochs=1):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A dataset that can be used for iteration.
  """
  filenames = get_filenames(is_training, data_dir)
  assert len(filenames)>0, (data_dir)
  print('\ngot {} tfrecord files\n'.format(len(filenames)))
  dataset = tf.data.TFRecordDataset(filenames,
                                    compression_type="",
                                    buffer_size=_SHUFFLE_BUFFER,
                                    num_parallel_reads=None)

  #if is_training:
  #  # Shuffle the input files
  #  dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

  # Convert to individual records.
  # cycle_length = 10 means 10 files will be read and deserialized in parallel.
  # This number is low enough to not cause too much contention on small systems
  # but high enough to provide the benefits of parallelization. You may want
  # to increase this number if you have a large number of CPU cores.
  #dataset = dataset.apply(tf.contrib.data.parallel_interleave(
  #    tf.data.TFRecordDataset, cycle_length=10))

  return resnet_run_loop.process_record_dataset(
      dataset, is_training, batch_size, _SHUFFLE_BUFFER, parse_pl_record, data_paras,
      num_epochs
  )


def get_synth_input_fn():
  return resnet_run_loop.get_synth_input_fn(
      _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS, _NUM_CLASSES)

def get_data_shapes_from_tfrecord(data_dir):
  global _DATA_PARAS

  batch_size = 1
  with tf.Graph().as_default():
    dataset = input_fn(True, data_dir, batch_size)
    iterator = dataset.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
      features, label = sess.run(iterator)

      for key in features:
        _DATA_PARAS[key] = features[key][0].shape

def get_data_meta_from_hdf5(data_dir):
  global _DATA_PARAS
  from block_data_prep_util import GlobalSubBaseBLOCK
  gsbb_load = GlobalSubBaseBLOCK()
  basen = os.path.basename(data_dir)
  dirn = os.path.dirname(os.path.dirname(data_dir))
  bxmh5_dir = os.path.join(dirn, 'ORG_bxmh5', basen)
  bxmh5_fns = glob.glob(os.path.join(bxmh5_dir,'*.bxmh5'))
  assert len(bxmh5_fns) > 0, (bxmh5_dir)
  bxmh5_fn = bxmh5_fns[0]
  gsbb_load.load_para_from_bxmh5(bxmh5_fn)

  _DATA_PARAS['dataset_name'] = DATASET_NAME
  _DATA_PARAS['sg_bm_extract_idx'] = gsbb_load.sg_bidxmaps_extract_idx
  _DATA_PARAS['flatten_bm_extract_idx'] = gsbb_load.flatten_bidxmaps_extract_idx
  _DATA_PARAS['global_step'] = gsbb_load.global_step
  _DATA_PARAS['global_stride'] = gsbb_load.global_stride
  _DATA_PARAS['sub_block_stride_candis'] = gsbb_load.sub_block_stride_candis
  _DATA_PARAS['sub_block_step_candis'] = gsbb_load.sub_block_step_candis
  _DATA_PARAS['flatbxmap_max_nearest_num'] = gsbb_load.flatbxmap_max_nearest_num


###############################################################################
# Running the model
###############################################################################
class ModelnetModel(resnet_model.Model):
  """Model class with appropriate defaults for Modelnet data."""

  def __init__(self, model_flag, resnet_size, data_format=None, num_classes=_NUM_CLASSES,
               resnet_version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE, data_paras={}):
    """These are the parameters that work for Modelnet data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      bottleneck = False
      final_size = 512
    else:
      bottleneck = True
      final_size = 2048

    super(ModelnetModel, self).__init__(
        model_flag = model_flag,
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        final_size=final_size,
        resnet_version=resnet_version,
        data_format=data_format,
        dtype=dtype,
        data_paras=data_paras
    )


def _get_block_sizes(resnet_size):
  """Retrieve the size of each block_layer in the ResNet model.

  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.

  Args:
    resnet_size: The number of convolutional layers needed in the model.

  Returns:
    A list of block sizes to use in building the model.

  Raises:
    KeyError: if invalid resnet_size is received.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)


def modelnet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""
  learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=256,
      num_images=_NUM_IMAGES['train'], boundary_epochs=[30, 60, 80, 90],
      decay_rates=[1, 0.1, 0.01, 0.001, 1e-4])

  return resnet_run_loop.resnet_model_fn(
      model_flag=params['model_flag'],
      features=features,
      labels=labels,
      mode=mode,
      model_class=ModelnetModel,
      resnet_size=params['resnet_size'],
      weight_decay=1e-4,
      learning_rate_fn=learning_rate_fn,
      momentum=0.9,
      data_format=params['data_format'],
      resnet_version=params['resnet_version'],
      loss_scale=params['loss_scale'],
      loss_filter_fn=None,
      dtype=params['dtype'],
      data_paras=params['data_paras']
  )


def define_modelnet_flags():
  global _DATA_PARAS
  _DATA_PARAS = {}

  resnet_run_loop.define_resnet_flags(
      resnet_size_choices=['18', '34', '50', '101', '152', '200'])
  flags.adopt_module_key_flags(resnet_run_loop)
  flags.DEFINE_string('model_flag', '3Vm','')
  flags.DEFINE_string('bxm_folder', '4096_mgs1_gs2_2d2_fmn1444_mvp1-3200_1024_48_1-18_24_56_56-0d1_0d2_0d6-0d0_0d1_0d4-pd3-neg-3M1','')
  data_dir = os.path.join(DATA_DIR, '/home/z/Research/dynamic_pointnet/data/MODELNET40__H5F/ORG_tfrecord/4096_mgs1_gs2_2-mbf-neg_fmn14_mvp1-1024_240_1-64_27_256-0d2_0d4-0d1_0d2-pd3-2M2pp')
  flags_core.set_defaults(train_epochs=100,
                          data_dir=data_dir,
                          batch_size=2)
  get_data_shapes_from_tfrecord(data_dir)
  get_data_meta_from_hdf5(data_dir)


def run_imagenet(flags_obj):
  """Run ResNet ImageNet training and eval loop.

  Args:
    flags_obj: An object containing parsed flag values.
  """
  input_function = (flags_obj.use_synthetic_data and get_synth_input_fn()
                    or input_fn)
  resnet_run_loop.resnet_main(
      flags_obj, modelnet_model_fn, input_function, DATASET_NAME, _DATA_PARAS)


def main(_):
  run_imagenet(flags.FLAGS)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  define_modelnet_flags()
  absl_app.run(main)