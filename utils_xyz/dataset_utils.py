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


def encode_to_raw(in_tensor):
  # in_tensor: a tensor with any dtype
  character_lookup = tf.constant([chr(i) for i in range(256)])
  encoded_string = tf.reduce_join(
      tf.gather(character_lookup,
                tf.cast(tf.bitcast(in_tensor, tf.uint8), tf.int32)))
  org_dtype = in_tensor.dtype
  org_shape = in_tensor.shape
  return encoded_string, org_dtype, org_shape


def decode_raw(encoded_string, org_dtype, org_shape):
  out_tensor = tf.reshape(tf.decode_raw(encoded_string, org_dtype),
                          org_shape) # Shape information is lost
  return out_tensor


def test_encode_raw():
  with tf.Graph().as_default():
    starting_dtype = tf.float32
    starting_tensor = tf.random_normal(shape=[10, 10], stddev=1e5,
                                      dtype=starting_dtype)

    encoded_string, org_dtype, org_shape = encode_to_raw(starting_tensor)
    back_to_tensor = decode_raw(encoded_string, org_dtype, org_shape)
    with tf.Session() as session:
      before, after = session.run([starting_tensor, back_to_tensor])
      print(before - after)


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



def process_pl_to_uint16( points, point_idxs ):
  # Preprocess point cloud and transfer all to uint16
  # points: [num_gblocks, num_point, channels]
  # point_idxs: {'xyz':[0,1,2], 'nxnynz':[3,4,5]}

  assert points.ndim == 3
  for ele in point_idxs:
    idx = point_idxs[ele]
    if ele == 'xyz':
      # set min as 0, make all positive
      min_xyz = points[:,:,idx].min(axis=-1, keepdims=True)
      points[:,:,idx] -= min_xyz

      # from m to mm
      points[:,:,idx] *= 1000

    elif ele=='nxnynz':
      points[:,:,idx] += 1
      points[:,:,idx] *= 10000
    else:
      raise NotImplementedError

  # check scope inside uint16
  max_data = points.max()
  min_data = points.min()
  assert max_data <= 65535
  assert min_data >= 0

  points = points.astype( np.uint16 )

  # reshape to make image like array
  data_ele_idxs = {}
  for i, ele in enumerate(point_idxs):
    assert len(point_idxs[ele]) == 3
    data_ele_idxs[ele] = point_idxs[ele][0]//3
  points = points.reshape( [points.shape[0], points.shape[1], -1, 3] )
  return points, data_ele_idxs


def pointcloud_to_tfexample(points, object_label):
    assert points.dtype == np.float32
    points_bin = points.tobytes()
    points_shape_bin = np.array(points.shape, np.int32).tobytes()
    example = tf.train.Example(features=tf.train.Features(feature={
      'points/encoded': bytes_feature(points_bin),
      'points/shape': bytes_feature(points_shape_bin),
      'object/label': int64_feature(object_label)
    }))
    return example

def write_pl_dataset(tfrecord_writer, datasource_name, points, point_idxs, object_labels, offset=0, pl_spbin_filename=None ):
  # points: [num_gblocks, num_point, channels], np.float32
  # point_idxs: {'xyz':[0,1,2], 'nxnynz':[3,4,5]}

  num_gblocks = points.shape[0]
  for j in range(num_gblocks):
    if object_labels == None:
      object_label = None
    else:
      object_label = object_labels[j]
    example = pointcloud_to_tfexample(points[j], object_label)
    tfrecord_writer.write(example.SerializeToString())

  return offset + num_gblocks

def bxmap_to_tfexample(sg_all_bidxmaps, bidxmaps_flat, fmap_neighbor_idis):
  assert sg_all_bidxmaps.dtype == np.int32
  assert bidxmaps_flat.dtype == np.int32
  assert fmap_neighbor_idis.dtype == np.float32

  sg_all_bidxmaps_bin = sg_all_bidxmaps.tobytes()
  bidxmaps_flat_bin = bidxmaps_flat.tobytes()
  fmap_neighbor_idis_bin = fmap_neighbor_idis.tobytes()
  example = tf.train.Example(features=tf.train.Features(feature={
    'sg_all_bidxmaps': bytes_feature(sg_all_bidxmaps_bin),
    'bidxmaps_flat': bytes_feature(bidxmaps_flat_bin),
    'fmap_neighbor_idis': bytes_feature(fmap_neighbor_idis_bin)
  }))
  return example

def write_bxmap_dataset(bxm_tfrecord_writer, datasource_name, sg_all_bidxmaps, bidxmaps_flat, fmap_neighbor_idis):
  num_gblocks = sg_all_bidxmaps.shape[0]
  for j in range(num_gblocks):
    example = bxmap_to_tfexample(sg_all_bidxmaps[j], bidxmaps_flat[j], fmap_neighbor_idis[j])
    bxm_tfrecord_writer.write(example.SerializeToString())


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


def read_from_tfrecord(filenames):
  tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
  reader = tf.TFRecordReader()
  tfrecord_key, tfrecord_serialized = reader.read(tfrecord_file_queue)

  tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                                              features={
                                                'datasource_name': tf.FixedLenFeature([], tf.string),
                                                'pl/data/xyz': tf.FixedLenFeature([], tf.string)
                                              },
                                              name = 'features')
  datasource_name = tf.cast(tfrecord_features['datasource_name'], tf.string)
  xyz = tf.decode_raw(tfrecord_features['pl/data/xyz'], tf.uint16)

  return datasource_name, xyz, tfrecord_features

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



def read_from_spbin(filenames):
    tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)

    # label and image are stored as bytes but could be stored as
    # int64 or float64 values in a serialized tf.Example protobuf.
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                        features={
                            'object/label': tf.FixedLenFeature([], tf.int64),
                            'points/shape': tf.FixedLenFeature([], tf.string),
                            'points/encoded': tf.FixedLenFeature([], tf.string),
                        }, name='features')
    points = tf.decode_raw(tfrecord_features['points/encoded'], tf.float32)
    shape = tf.decode_raw(tfrecord_features['points/shape'], tf.int32)
    # the image tensor is flattened out, so we have to reconstruct the shape
    points = tf.reshape(points, shape)
    object_label = tfrecord_features['object/label']
    return points, object_label


def read_spbin(filename=None):
  if filename == None:
    filename = '/home/z/Research/dynamic_pointnet/data/MODELNET40__H5F/ORG_sph5/4096_mgs1_gs2_2-mbf-neg/airplane_0001.tfrecord'
  points_, object_label_ = read_from_spbin([filename])

  with tf.Session() as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord)
      points, object_label = sess.run([points_, object_label_])
      coord.request_stop()
      coord.join(threads)
  print(points)
  return points, object_label


if __name__ == '__main__':
  #test_encode_raw()
  read_spbin()

