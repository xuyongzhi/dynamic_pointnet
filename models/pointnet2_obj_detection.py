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
sys.path.append(os.path.join(BASE_DIR, '../utils_xyz'))
import tensorflow as tf
import numpy as np
from config import cfg
import numpy.random as npr
import tf_util
from bbox_transform import bbox_transform
from pointnet_util import pointnet_sa_module, pointnet_fp_module
from sklearn.metrics.pairwise import euclidean_distances


def placeholder_inputs(batch_size, num_point,num_channel=3, num_regression=7):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_channel))
    labels_pl = tf.placeholder(tf.float32, shape=(batch_size, None, num_regression+1))   # label = category, l, w ,h, alpha, x, y, z
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
    num_regression = cfg.TRAIN.NUM_REGRESSION  # 7 = l,w,h,theta,x,y,z
    # Layer 1
    # [8,1024,3] [8,1024,64] [8,1024,32]
    # Note: the important tuning parameters  radius_l* = 1 m
    radius_l1 = cfg.TRAIN.Radius_1
    radius_l2 = cfg.TRAIN.Radius_2
    radius_l3 = cfg.TRAIN.Radius_3
    radius_l4 = cfg.TRAIN.Radius_4
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=8192, radius= radius_l1, nsample=32, mlp=[32,32,64]   , mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=2048, radius= radius_l2, nsample=32, mlp=[64,64,128]  , mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=1024, radius= radius_l3, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
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


def get_loss(pred_class, pred_box, gt_box, smpw, xyz):
    """
    Input: pred_class: BxNx3xC
           pred_box: BxNx3x7
           gt_box: BxNx7
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
    assert xyz.shape[2] == 3
    NUM_BATCH = xyz.shape[0]
    A = cfg.TRAIN.NUM_ANCHORS
    loss_classification_all = 0.0
    loss_regression_all  = 0.0

    for n in range(0,NUM_BATCH):
        # N is the points number of xyz
        # all_alpha is the [N A] A anchors, reshape to [N*A 1]
        # dif_alpha is the angle substract with label
        # calculate the angle gap to select one anchor
        N  =  xyz[n,:,:].shape[0]
        CC =  xyz[n,:,:].shape[1]

        # estimate the central points distance, using broadcasting ops
        #distance = euclidean_distances(xyz[n,:,:], gt_box[n,:,5:8])
        temp_xyz = tf.reshape(xyz[n,:,:], (N,1,CC))
        distance = tf.sqrt(tf.reduce_sum(tf.square(tf.sbutract(temp_xyz, gt_box[n,:,5:8])),2))

        #distance = np.tile(distance, (1,A))
        distance = tf.tile(distance, (1,A))
        #distance = distance.reshape(-1,  label[n].shape(0)) # label[n].shape(0)	is the number of ground truth
        distance = tf.reshape(distance, (-1, gt_box[n,:,:].shape(0)))

        #labels = np.zeros(shape=(N*A,1))
        labels =  tf.fill([N*A, 1], -1)

        #labels.fill(-1)

        # decide positive and negative labels
        argmin_dist = distance.argmax(axis=1)
        min_dist    = distance[np.arange(distance.shape[0]),argmin_dist]
        gt_argmin_dist = distance.argmax(axis=0)
        gt_min_dist  = distance[gt_argmin_dist, np.arange(distance.shape[1])]
        gt_argmin_dist = np.where(distance == gt_min_dist)[0]

        all_alpha = np.tile(cfg.TRAIN.Alpha, (N,1))
        all_alpha = all_alpha.reshape(-1,1)
        dif_alpha = all_alpha - gt_box[n, argmin_dist, 4]


        # Deleting hard coding
        # arg_alpha0  = np.where(np.absolute(dif_alpha[:,0]) <  np.pi/4)  # alpha is 0
        # arg_alpha90 = np.where(np.absolute(dif_alpha[:,1]) <= np.pi/4) # alpha is 90
        # labels[np.intersect1d(arg_alpha0,gt_argmin_dist),0]  = 1 # alpha 0
        # labels[np.intersect1d(arg_alpha90,gt_argmin_dist),1] = 1 # alpha 90
        # labels[np.intersect1d(arg_alpha0 ,(min_dist < cfg.TRAIN.POSITIVE_CEN_DIST)),0] = 1
        # labels[np.intersect1d(arg_alpha90,(min_dist < cfg.TRAIN.POSITIVE_CEN_DIST)),1] = 1
        #labels[np.intersect1d(arg_alpha90 ,(min_dist > cfg.TRAIN.NEGATIVE_CEN_DIST)),0] = 0
        #labels[np.intersect1d(arg_alpha0, (min_dist > cfg.TRAIN.NEGATIVE_CEN_DIST)),1] = 0
        arg_alpha = np.where(np.absolute(dif_alpha[:0] < cfg.TRAIN.POSITIVE_ALPHA))[0]
        labels[np.intersect1d(arg_alpha, gt_argmin_dist)] = 1
        min_dist_positive_inds = np.where(min_dist < cfg.TRAIN.POSITIVE_CEN_DIST)[0]
        labels[np.intersect1d(arg_alpha, min_dist_positive_inds)] = 1
        min_dist_negative_inds = np.where(min_dist > cfg.TRAIN.NEGATIVE_CEN_DIST)[0]
        labels[np.intersect1d(arg_alpha, min_dist_negative_inds)] = 0

        num_positive_labels = int( cfg.TRAIN.RPN_FG_FRACTION*cfg.TRAIN.RPN_BATCHSIZE )
        # labels = labels.reshape(-1,1)
        positive_inds = np.where(labels == 1)[0]
        if len(positive_inds) > num_positive_labels:
		disable_inds = npr.choice(
			positive_inds, size = (len(positive_inds) - num_positive_labels), replace = False )
		labels[disable_inds] = -1

        num_negative_labels = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        negative_inds = np.where(labels == 0)[0]
        if len(negative_inds) > num_negative_labels:
            disable_inds = npr.choice(
			negative_inds, size = (len(negative_inds) - num_negative_labels), replace = False )
            labels[disable_inds] = -1
            # labels = labels.reshape(-1,A)


        box_targets = anch_boxes = gt_boxes = box_inside_weights = box_outside_weights = np.zeros((N*A,cfg.TRAIN.NUM_REGRESSION), dtype = np.float32 )
        anch_box[:,0] = cfg.TRAIN.Anchors[0,0]
        anch_box[:,1] = cfg.TRAIN.Anchors[0,1]
        anch_box[:,2] = cfg.TRAIN.Anchors[0,2]
        anch_box[:,3] = all_alpha
        all_xyz       = np.tile(xyz,(1,A))
        all_xyz       = all_xyz.reshape(-1, 3)
        anch_box[:,4] = all_xyz[:,0]
        anch_box[:,5] = all_xyz[:,1]
        anch_box[:,6] = all_xyz[:,2]

        # outside weights and inside_weigths

        box_inside_weights[labels==1,:]  = np.array(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)
        box_outside_weights[labels==1,:] = np.ones(1,cfg.TRAIN.NUM_REGRESSION)*1.0/cfg.TRAIN.RPN_BATCHSIZE
        box_outside_weights[labels==0,:] = np.ones(1,cfg.TRAIN.NUM_REGRESSION)*1.0/cfg.TRAIN.RPN_BATCHSIZE

        # calculate the box targets between anchors and ground truth
        assert anch_box.shape[1] == cfg.TRAIN.NUM_REGRESSION
        # assert gt_box.shape[1] == cfg.TRAIN.NUM_REGRESSION
        box_targets   = bbox_transform(anch_box, gt_box[argmin_dist,0:7])

        box_pred = tf.reshape(box_pred, [-1, cfg.TRAIN.NUM_REGRESSION])
        regresion_smooth = _smooth_l1(cfg.TRAIN.SIGMA, box_pred, box_targets, box_inside_weights, box_outside_weights)
        loss_regression  = tf.reduce_mean(tf.reduce_sum(regression_smooth, axis = 1))

        # classification loss
        pred_class = tf.reshape(pred_class, [-1,2])
        labels = tf.reshape(labels, [-1])
        pred_class = tf.reshape(tf.gather(pred_class, tf.where(tf.not_equal(labels,-1))),[-1, 2])
        labels = tf.reshape(tf.gather(labels, tf.where(tf.not_equal(labels, -1))), [-1])
        loss_classification = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw))

        loss_classification_all = loss_classification_all +  loss_classification
        loss_regression_all     = loss_regression_all + loss_regression

        # regression loss

    loss_all = (loss_classification_all + cfg.TRAIN.LAMBDA*loss_regression_all)/NUM_BATCH
    tf.summary.scalar('classification loss', loss_classification_all/NUM_BATCH)
    tf.summary.scalar('regression loss', loss_regression_all/NUM_BATCH)
    return loss_all


def _smmoth_l1(sigma ,box_pred, box_targets, box_inside_weights, box_outside_weights):
    """
        loss = bbox_outside_weights*smoothL1(inside_weights * (box_pred - box_targets))
        smoothL1(x) = 0.5*(sigma*x)^2, if |x| < 1 / sigma^2
                       |x| - 0.5 / sigma^2, otherwise
    """
    sigma2 = sigma * sigma
    inside_mul = tf.multiply(box_inside_weights, tf.substract(box_pred, box_targets))

    smooth_l1_sign    = tf.cast(tf.less(tf.abs(inside_mul), 1.0/sigma2), tf.float32)
    smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul,inside_mul), 0.5*sigma2)
    smooth_l1_option2 = tf.substract(tf.abs(inside_mul), 0.5*sigma2)

    smooth_l1_result  = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                               tf.multiply(smooth_l1_option2, tf.abs(tf.substract(smooth_l1_sign, 1.0))))

    outside_mul   = tf.multiply(box_outside_weights, smooth_l1_result)

    return outside_mul




if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        num_class = cfg.TRAIN.NUM_CLASS
        net, _ , _ , _ = get_model(inputs, tf.constant(True), num_class)
        print(net)
