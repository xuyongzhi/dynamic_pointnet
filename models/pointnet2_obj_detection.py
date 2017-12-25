# --------------------------------------------------------
# 3D object detection model for point cloud
# Licensed under The MIT License [see LICENSE for details]
# Created on 17/12/2017
# Modified by Xuesong Li
# --------------------------------------------------------

"""
Descrption:
"""

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../config'))
import tensorflow as tf
import numpy as np
from config import cfg
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module
from sklearn.metrics.pairwise import euclidean_distances


def placeholder_inputs(batch_size, num_point,num_channel=3):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_channel))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl, smpws_pl


def get_model(point_cloud, is_training, num_class, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = None
    end_points['l0_xyz'] = l0_xyz

    # for each point (x,y,z) p
    #   generate 3 3D anchors at p
    #   then apply the predict box deltas to each anchor boxes
    #   calculate the box deltas beween 3D anchor bounding box and ground truth
    #
    num_3d_anchors = cfg.TRAIN.NUM_ANCHORS
    num_regression = cfg.TRAIN.NUM_REGRESSION  # 7 = x,y,z,l,w,h,theta
    # Layer 1
    # [8,1024,3] [8,1024,64] [8,1024,32]
    # Note: the important tuning parameters  radius_l* = 1 m
    radius_l1 = cfg.TRAIN.Radius_1
    radius_l2 = cfg.TRAIN.Radius_2
    radius_l3 = cfg.TRAIN.Radius_3
    radius_l4 = cfg.TRAIN.Radius_4
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=8192, radius= radius_l1, nsample=32, mlp=[32,32,64]   , mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=2048, radius= radius_l2, nsample=32, mlp=[64,64,128]  , mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoin =1024, radius= radius_l3, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint= 512, radius= radius_l4, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # Feature Propagation layers
    #l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    #l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer2')
    #l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer3')
    #l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l4_points, 512, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net_class = tf_util.conv1d(net, num_3d_anchors*num_class     , 1 , padding='VALID', activation_fn=None, scope='fc2') # outputing the classification for every point
    net_boxes = tf_util.conv1d(net, num_3d_anchors*num_regression, 1 , padding='VALID', activation_fn=None, scope='fc3') # outputing the 3D bounding boxes

    return end_points, net_class, net_boxes, l4_xyz


def get_loss(pred_class, pred_box, label, smpw, xyz):
    """
    Input: pred_class: BxNx3xC
           pred_box: BxNx3x7
           label: BxNx7
	       smpw: BxN
    Description: similar to faster rcnn, overlaps between prediction boxes and ground truth boxes are estimated to decide point labels and further calucuate the box_targets
    --> preparing all anchors in every space point --> calculating overlaps bewteen anchor boxes and bounding boxes -->  anchors whose overlaps are over a certain threshold are regarded as positive labels, the rest are negative labels
    --> calculate the box_targets between anchors and ground truth --> prepare the outside_weight and inside_weight -->  then write the loss function
    """
    # prepare all possible anchors for all points, xyz are point central+all the
    # anchors, we estimate the birdview of 3D bounding boxes to accelerate the
    # program, coordinate systems x: forward, y: left, z: up, so for the bird
    # view, we use x and y
    # xyz = [batch, point, 3]
    # add all acnhors(1,k,4) to
    # every point_xyz (N,1,4)
    # shift the anchors(N, K, 4) to (N*K, 4)

    # A = cfg.TRAIN.NUM_ANCHORS
    # shifts = np.vstack(xyz[0,:,0],xyz[0,:,1],xyz[0,:,0],xyz[0,:,1]).transpose()
    # K = shifts.shape[0]
    # all_anchors =  cfg.TRAIN.Anchor_bv.reshape(1, A, 4) + shifts.reshape(k,1,4)
    # all_anchors = all_anchors.reshape(K*A, 4)
    # calculating the overlap between anchors and ground truth
    NUM_BATCH = xyz.shape[0]
    A = cfg.TRAIN.NUM_ANCHORS

    for n in range(0,NUM_BATCH)
        # N is the points number of xyz
        # all_alpha is the [N 2] 2 anchors
        # dif_alpha is the angle substract with label
        # calculate the angle gap to select one anchor
        N =  xyz[n].shape[1]
        all_alpha = np.tile(cfg.TRAIN.Alpha, (N,1))
        dif_alpha = all_alpha - label[n].alpha

        # estimate the central points distance
        distance = euclidean_distances(xyz[n],label[n].xyz)

        labels = np.zeros(shape=(N, A))
        labels.fill(-1)
        # decide positive and negative labels

        # calculate the box targets between anchors and ground truth

        # outside weights and inside_weigths

        # classification loss

        # regression loss


    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)
