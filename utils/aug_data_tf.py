# xyz June 2018

import tensorflow as tf
import os, sys

BASE_DIR = os.path.abspath(__file__)
sys.path.append(BASE_DIR)

from geometric_tf_util import *
import numpy as np



def random_rotate():
  '''
      Rotate the whole object by a random angle 3D
  '''

  angles = tf.random_uniform([3], dtype=tf.float32) * 2 * np.pi
  R = tf_EulerRotate(angles, 'zxy')
  return R

def random_scaling(min_scale=0.8, max_scale=1.25):
  scales = tf.random_uniform([3], minval=min_scale, maxval=max_scale)
  S = tf.concat([[[scales[0], 0, 0],
                  [0, scales[1], 0],
                  [0, 0, scales[2]] ]], 0)
  return S

def random_shift(shift_range=0.1):
  shifts = tf.random_uniform([1,3], minval=-shift_range, maxval=shift_range)
  return shifts

def random_jitter(points_shape, sigma=0.01, clip=0.05):
  jitter = tf.random_normal(points_shape, mean=0.0, stddev=sigma)
  jitter = tf.clip_by_value(jitter, -clip, clip)
  return jitter

def aug_all(points, data_idxs, \
            aug_items=['rotation', 'scaling', 'shifts', 'jitter']):
  '''
      points: [n,3] 'xyz'
              [n,6] 'xyznxnynz'
  '''
  point_num = points.shape[0].value
  channels = points.shape[1].value
  assert (data_idxs['xyz'] == np.array([0,1,2])).all()
  if channels>3:
    assert (data_idxs['nxnynz'] == np.array([3,4,5])).all()
  assert channels==3 or channels==6

  augs = {}
  if 'rotation' in aug_items:
    R = random_rotate()
    augs['R'] = R
  if 'scaling' in aug_items:
    S = random_scaling()
    augs['S'] = S

  if 'rotation' in aug_items and 'scaling' in aug_items:
    RS = tf.matmul(R, S)
  elif 'rotation' in aug_items:
    RS = R
  elif 'scaling' in aug_items:
    RS = S
  else:
    RS = None
  if RS!=None:
    points_xyz = tf.matmul(points[:,0:3], RS)
    augs['RS'] = RS
  else:
    points_xyz = points[:,0:3]


  if 'shifts' in aug_items:
    shifts = random_shift()
    points_xyz += shifts
    augs['shifts'] = shifts

  if 'jitter' in aug_items:
    jitter = random_jitter((point_num,3))
    points_xyz += jitter
    augs['jitter'] = jitter

  if channels==3:
    points = points_xyz
  elif channels==6:
    if 'rotation' in aug_items:
      points_normal = tf.matmul(points[:,3:6], R)
    else:
      points_normal = points[:,3:6]
    points = tf.concat([points_xyz, points_normal], -1)

  return points, augs

def aug_bottom_center(bottom_center_batch, augs):
  import pdb; pdb.set_trace()  # XXX BREAKPOINT
  assert len(bottom_center_batch.shape) == 3
  assert bottom_center_batch.shape[-1].value == 6

  batch_size = bottom_center_batch.shape[0].value
  n0 = bottom_center_batch.shape[1].value
  bottom_center_batch = tf.reshape(bottom_center_batch, [batch_size,-1,3])
  if 'RS' in augs:
    bottom_center_batch = tf.matmul(bottom_center_batch, augs['RS'])
  if 'shifts' in augs:
    bottom_center_batch += augs['shifts']
  bottom_center_batch = tf.reshape(bottom_center_batch, [batch_size,n0,6])
  return bottom_center_batch

def get_aug_items(aug_type):
  if aug_type == 'all':
    aug_items = ['rotation', 'scaling', 'shifts', 'jitter']
  elif aug_type == 'r':
    aug_items = ['rotation']
  elif aug_type == 's':
    aug_items = ['scaling']
  elif aug_type == 'f':
    aug_items = ['shifts']
  elif aug_type == 'j':
    aug_items = ['jitter']
  else:
    raise NotImplementedError
  return aug_items

def aug_data(points, b_bottom_centers_mm, aug_type, data_idxs):
  if aug_type=='none':
    return points, b_bottom_centers_mm, None
  else:
    aug_items = get_aug_items(aug_type)
    points, augs = aug_all(points, data_idxs, aug_items)
    return points, b_bottom_centers_mm, augs


