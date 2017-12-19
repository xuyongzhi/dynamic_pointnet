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

#
# Training options
#

__C.TRAIN = edict()

# number of 3D anchors
__C.TRAIN.NUM_ANCHORS = 3

# number of regression parameters,  7 = x,y,z,l,w,h,theta
__C.TRAIN.NUM_REGRESSION = 7

# radius for grouping
__C.TRAIN.Radius_1 = 1
__C.TRAIN.Radius_2 = 1
__C.TRAIN.Radius_3 = 1
__C.TRAIN.Radius_4 = 1


# 3D anchor size, the same size (l=3.9m, w=1.6m, h=1.7m) but with different
# orientation(pi = 0, pi/4, pi/2)
__C.TRAIN.Anchors = np.array([3.9 1.9 1.7])


# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = False

# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7

# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3

# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5

# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256

# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7

# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 1000

# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 200

# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TRAIN.RPN_MIN_SIZE = 16

# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

#
# Testing options
#
__C.TEST = edict()

# IoU >= this threshold)
__C.TEST.NMS = 0.3

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7

## Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 1000

## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 200

# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 16

