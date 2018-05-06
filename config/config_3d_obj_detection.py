# --------------------------------------------------------
# configuration file in 3D object detection for point cloud
# Licensed under The MIT License [see LICENSE for details]
# Created on 18/12/2017
# Modified by Xuesong Li
# -----------------------------------------------------

"""
Description
  This file specifies default config options for 3D object detection. Instead of changing value here,
  you can write a config file in yaml and load it to override default options.

"""

import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

# DEBUG = False
#
# Training options
#

# __C.= edict()

# number of 3D anchors
__C.NUM_ANCHORS = 2

# number of classification
__C.NUM_CLASSES = 1  ## background and vehicle, every anchor just have one classes, start training from simple situation

# number of channels
__C.NUM_CHANNELS = 3 ## xyz

# number of evaluation
__C.EVALUATION_NUM = 700 # 1000

# number of regression parameters,  7 = x, y, l, w, h, alpha, (ignore z, for z
# is too difficulat to predict)
__C.NUM_REGRESSION = 6

# radius for grouping
__C.Radius_1 = 0.4
__C.Radius_2 = 0.8
__C.Radius_3 = 1.6
__C.Radius_4 = 2.4
__C.Radius_5 = 3.2
__C.Radius_6 = 2.6

# 3D anchor size, the same size (l=3.9m, w=1.6m, h=1.7m) but with different
# orientation alpha (pi = 0, pi/4, pi/2)
l=3.9
w=1.6
h=1.7
__C.Anchors = np.array([l, w])
__C.Alpha = np.array([[0], [np.pi/2]])  ## if 2 anchor is not enought, change it to 4, [0, np.pi/4, np.pi/2, np.pi*3/4]
__C.Anchor_bv = np.array([[l/2,  w/2,  -l/2,  -w/2 ] , [w/2,  l/2,  -w/2,  -l/2]])   ## [frowart_left back_right]


## the number of anchor coordinate lists is up to the number of anchors
__C.Anchor_coordinate_lists = np.array([[[-w/2.0, l/2.0],[w/2.0, l/2.0],[w/2.0, -l/2.0],[-w/2.0, -l/2.0], [-w/2.0, l/2.0]],
                                             [ [-l/2.0, w/2.0],[l/2.0, w/2.0],[l/2.0, -w/2.0],[-l/2.0, -w/2.0], [-l/2.0, w/2.0] ]])
# Use horizontal flipper point cloud, a way of data augmentation
__C.USE_FLIPPED = True

# Deprecated (inside weights)
__C.BBOX_INSIDE_WEIGHTS = np.ones((1,__C.NUM_REGRESSION))*1.0 # (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


__C.DEBUG =False  # True

__C.SMALL_addon_for_BCE = 1e-6  ## BCE: binary classification entropy

__C.SIGMA = 2.0
# IOU >= thresh: positive example

__C.LAMBDA = 0.6

__C.ALPHA = 1.5

__C.BETA = 1.0

#__C.RPN_POSITIVE_OVERLAP = 0.7

# IOU < thresh: negative example
#__C.RPN_NEGATIVE_OVERLAP = 0.3

# Alpha thershold between anchors and ground truth
__C.POSITIVE_ALPHA = np.pi/4

# The overlap to tell positive and negative samples
__C.POSITIVE_THRESHOLD = 0.7

__C.NEGATIVE_THRESHOLD = 0.2

# Max number of foreground examples
__C.RPN_FG_FRACTION = 0.5

# Total number of examples
__C.RPN_BATCHSIZE = 64

# NMS threshold used on RPN proposals
__C.RPN_THRESH = 0.7

# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.RPN_PRE_NMS_TOP_N = 1000

# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.RPN_POST_NMS_TOP_N = 200

# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.RPN_MIN_SIZE = 16

# Deprecated (outside weights)
__C.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

