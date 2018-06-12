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

def aug_all(points):
  '''
      points: [n,3] 'xyz'
              [n,6] 'xyznxnynz'
  '''
  assert len(points.shape) == 2
  channels = points.shape[1].value
  assert channels == 3 or channels == 6

  R = random_rotate()
  S = random_scaling()
  RS = tf.matmul(R, S)

  points_xyz = tf.matmul(points[:,0:3], RS)
  if channels==6:
    points_normal = tf.matmul(points[:,3:6], R)
    points = tf.concat([points_xyz, points_normal], -1)
  else:
    points = points_xyz

  return points, R, S

def aug_data(points, aug_type='none'):
  if aug_type=='none':
    return points, None, None
  elif aug_type=='all':
    points, R, S = aug_all(points)
    return points, R, S
  else:
    raise NotImplementedError

