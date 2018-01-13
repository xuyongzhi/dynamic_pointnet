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

DEBUG = False
#
# Training options
#

__C.TRAIN = edict()

# number of 3D anchors
__C.TRAIN.NUM_ANCHORS = 2

# number of classification
__C.TRAIN.NUM_CLASSES = 2  ## background and vehicle, start training from simple situation

# number of channels
__C.TRAIN.NUM_CHANNELS = 3 ## xyz

# number of evaluation
__C.TRAIN.EVALUATION_NUM = 1000

# number of regression parameters,  7 = x,y,z,l,w,h,theta
__C.TRAIN.NUM_REGRESSION = 7

# radius for grouping
__C.TRAIN.Radius_1 = 0.4
__C.TRAIN.Radius_2 = 0.8
__C.TRAIN.Radius_3 = 1.2
__C.TRAIN.Radius_4 = 1.6


# 3D anchor size, the same size (l=3.9m, w=1.6m, h=1.7m) but with different
# orientation alpha (pi = 0, pi/4, pi/2)
l=3.9
w=1.6
h=1.7
__C.TRAIN.Anchors = np.array([l, w, h])
__C.TRAIN.Alpha = np.array([[0], [np.pi/2]])  ## if 2 anchor is not enought, change it to 4, [0, np.pi/4, np.pi/2, np.pi*3/4]
__C.TRAIN.Anchor_bv = np.array([[l/2,  w/2,  -l/2,  -w/2 ] , [w/2,  l/2,  -w/2,  -l/2]])   ## [frowart_left back_right]

# Use horizontal flipper point cloud, a way of data augmentation
__C.TRAIN.USE_FLIPPED = True

# Deprecated (inside weights)
__C.TRAIN.BBOX_INSIDE_WEIGHTS = np.ones((1,__C.TRAIN.NUM_REGRESSION))*1.0 # (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


__C.TRAIN.SIGMA = 2.0
# IOU >= thresh: positive example

__C.TRAIN.LAMBDA = 1.0

#__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7

# IOU < thresh: negative example
#__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3

# Alpha thershold between anchors and ground truth
__C.TRAIN.POSITIVE_ALPHA = np.pi/4

# Distance between central points
__C.TRAIN.NEGATIVE_CEN_DIST = 0.88

__C.TRAIN.POSITIVE_CEN_DIST = 0.22

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

# number of 3D anchors
__C.TEST.NUM_ANCHORS = 3

# number of classification
__C.TEST.NUM_CLASS = 2  ## background and vehicle, training from simple situation

# number of regression parameters,  7 = x,y,z,l,w,h,theta
__C.TEST.NUM_REGRESSION = 7

# radius for grouping
__C.TEST.Radius_1 = 1
__C.TEST.Radius_2 = 1
__C.TEST.Radius_3 = 1
__C.TEST.Radius_4 = 1

# 3D anchor size, the same size (l=3.9m, w=1.6m, h=1.7m) but with different
# orientation alpha (pi = 0, pi/4, pi/2)
__C.TEST.Anchors = np.array([l, w, h])
__C.TEST.Alpha = np.array([0,np.pi/2])

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

