# xyz June 2018

import tensorflow as tf
import os, sys

BASE_DIR = os.path.abspath(__file__)
sys.path.append(BASE_DIR)

from geometric_tf_util import *
import numpy as np



def random_rotate(points):
  ''' points: [n,3] 'xyz'
              [n,6] 'xyznxnynz'
  '''
  assert len(points.shape) == 2
  channels = points.shape[1].value
  assert channels == 3 or channels == 6

  angles = tf.random_uniform([3], dtype=tf.float32) * 2 * np.pi
  R = tf_EulerRotate(angles, 'zxy')
  if channels==6:
    points = tf.reshape(points, [-1,3])
  points = tf.matmul(points, R)
  if channels==6:
    points = tf.reshape(points, [-1,6])
  return points, R

def aug_data(points):
  points, R = random_rotate(points)
  return points, R

