# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
import numpy as np

from six.moves import urllib
import tensorflow as tf

LABELS_FILENAME = 'labels.txt'


def int64_feature(values):
  """Returns a TF-Feature of int64s.

  Args:
    values: A scalar or list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
  """Returns a TF-Feature of bytes.

  Args:
    values: A string.

  Returns:
    A TF-Feature.
  """
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
  """Returns a TF-Feature of floats.

  Args:
    values: A scalar of list of values.

  Returns:
    A TF-Feature.
  """
  if not isinstance(values, (tuple, list)):
    values = [values]
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/class/label': int64_feature(class_id),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))

def pointcloud_to_tfexample(datasource_name, data_str, num_point, data_ele_idxs, object_classid ):
    example = tf.train.Example(features=tf.train.Features(feature={
      'datasource_name': bytes_feature(datasource_name),
      'pl/class/label': int64_feature(object_classid),
      'pl/data/encoded': bytes_feature(data_str),
      'pl/data/num_point': int64_feature(num_point),
      'pl/data/xyz': int64_feature(data_ele_idxs['xyz']),
      'pl/data/nxnynz': int64_feature(data_ele_idxs['nxnynz'])
    }))
    return example

def process_pl_to_uint16( datas, data_idxs ):
  # Preprocess point cloud and transfer all to uint16
  # datas: [num_gblocks, num_point, channels]
  # data_idxs: {'xyz':[0,1,2], 'nxnynz':[3,4,5]}

  assert datas.ndim == 3
  for ele in data_idxs:
    idx = data_idxs[ele]
    if ele == 'xyz':
      # set min as 0, make all positive
      min_xyz = datas[:,:,idx].min(axis=-1, keepdims=True)
      datas[:,:,idx] -= min_xyz

      # from m to mm
      datas[:,:,idx] *= 1000

    elif ele=='nxnynz':
      datas[:,:,idx] += 1
      datas[:,:,idx] *= 10000
    else:
      raise NotImplementedError

  # check scope inside uint16
  max_data = datas.max()
  min_data = datas.min()
  assert max_data <= 65535
  assert min_data >= 0

  datas = datas.astype( np.uint16 )

  # reshape to make image like array
  data_ele_idxs = {}
  for i, ele in enumerate(data_idxs):
    assert len(data_idxs[ele]) == 3
    data_ele_idxs[ele] = data_idxs[ele][0]//3
  datas = datas.reshape( [datas.shape[0], datas.shape[1], -1, 3] )
  return datas, data_ele_idxs


def write_pl_dataset(tfrecord_writer, datasource_name, datas, data_idxs, object_labels, offset=0 ):
  datas, data_ele_idxs = process_pl_to_uint16( datas, data_idxs )
  IsCheck = True

  with tf.Graph().as_default():
    pl_placeholder = tf.placeholder(dtype=tf.uint16)
    encoded_pl = tf.image.encode_png(pl_placeholder)
    pl_str_check = tf.placeholder(dtype=tf.string)
    pl_check = tf.image.decode_png(pl_str_check, dtype=tf.uint16)

    with tf.Session() as sess:
      num_gblocks = datas.shape[0]
      for j in range(num_gblocks):
        pl_str = sess.run(encoded_pl,
                          feed_dict={pl_placeholder: datas[j]} )

        if IsCheck:
          data_check = sess.run(pl_check,
                                feed_dict={pl_str_check:pl_str})
          assert  np.sum(data_check != datas[j])==0, ("decode check failed")

        if object_labels == None:
          object_classid = None
        else:
          object_classid = object_labels[j]
        example = pointcloud_to_tfexample(datasource_name, pl_str, datas.shape[1], data_ele_idxs, object_classid)
        tfrecord_writer.write(example.SerializeToString())

  return offset + num_gblocks


def download_and_uncompress_tarball(tarball_url, dataset_dir):
  """Downloads the `tarball_url` and uncompresses it locally.

  Args:
    tarball_url: The URL of a tarball file.
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = tarball_url.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)

  def _progress(count, block_size, total_size):
    sys.stdout.write('\r>> Downloading %s %.1f%%' % (
        filename, float(count * block_size) / float(total_size) * 100.0))
    sys.stdout.flush()
  filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
  print()
  statinfo = os.stat(filepath)
  print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
  """Writes a file with the list of class names.

  Args:
    labels_to_class_names: A map of (integer) labels to class names.
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'w') as f:
    for label in labels_to_class_names:
      class_name = labels_to_class_names[label]
      f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
  """Specifies whether or not the dataset directory contains a label map file.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    `True` if the labels file exists and `False` otherwise.
  """
  return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
  """Reads the labels file and returns a mapping from ID to class name.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    A map from a label (integer) to class name.
  """
  labels_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(labels_filename, 'rb') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    labels_to_class_names[int(line[:index])] = line[index+1:]
  return labels_to_class_names
