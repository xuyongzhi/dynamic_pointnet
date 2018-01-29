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
from soft_cross_entropy import softmaxloss



ISDEBUG = cfg.TRAIN.DEBUG


def placeholder_inputs(batch_size, num_point,num_channel=3, num_regression=7):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, num_channel))
    gt_box_pl = tf.placeholder(tf.float32, shape=(batch_size, None, num_regression+1))   # label = category, l, w ,h, alpha, x, y, z
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point))
    return pointclouds_pl, gt_box_pl, smpws_pl


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
    radius_l5 = cfg.TRAIN.Radius_5
    radius_l6 = cfg.TRAIN.Radius_6




    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint= 16384,radius= radius_l1, nsample=32, mlp=[32,32,64]   , mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint= 8192, radius= radius_l2, nsample=32, mlp=[64,64,128]  , mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint= 4096, radius= radius_l3, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint= 2048, radius= radius_l4, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')
    l5_xyz, l5_points, l5_indices = pointnet_sa_module(l4_xyz, l4_points, npoint= 1024, radius= radius_l5, nsample=32, mlp=[512,512,1024], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer5')
    l6_xyz, l6_points, l6_indices = pointnet_sa_module(l5_xyz, l5_points, npoint=  512, radius= radius_l6, nsample=32, mlp=[1024,1024,2048], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer6')



    # Feature Propagation layers
    #l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    #l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer2')
    #l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer3')
    #l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128,128], is_training, bn_decay, scope='fa_layer4')

    # FC layers
    net = tf_util.conv1d(l6_points, 1024, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net_class = tf_util.conv1d(net, num_3d_anchors*num_class     , 1 , padding='VALID', activation_fn=None, scope='fc2') # outputing the classification for every point
    net_boxes = tf_util.conv1d(net, num_3d_anchors*num_regression, 1 , padding='VALID', activation_fn=None, scope='fc3') # outputing the 3D bounding boxes

    return end_points, net_class, net_boxes, l6_xyz


def get_loss(batch_size, pred_class, pred_box, gt_box, smpw, xyz):
    '''
    pred_class: batch * num_point * num_anchor * 2
    pred_box  : batch * num_point * num_anchor * 7
    gt_box    : batch * n * (num_regression + 1) # 1 is the categroy
    xyz       : batch * num_point * 3
    '''
    gt_box = tf.convert_to_tensor(gt_box, tf.float32)
    output_box_targets, output_box_inside_weights, output_box_outside_weights, output_gt_class, output_labels = \
             tf.py_func(region_proposal_loss, [pred_class, pred_box, gt_box, smpw, xyz], [tf.float32, tf.float32, tf.float32, tf.int32, tf.int32])

    # all_loss = tf.convert_to_tensor(all_loss, name = 'all_loss')
    # classification_loss = tf.convert_to_tensor(classification_loss, name = 'classification_loss')
    # regression_loss = tf.convert_to_tensor(regression_loss, name = 'regression_loss')
    # output_pred_box_one_batch = tf.convert_to_tensor(output_pred_box_one_batch, name = 'output_pred_box_one_batch')
    # output_box_targets        = tf.convert_to_tensor(output_box_targets,        name = 'output_box_targets')
    # output_box_inside_weights = tf.convert_to_tensor(output_box_inside_weights, name = 'output_box_inside_weights')
    # output_box_outside_weights= tf.convert_to_tensor(output_box_outside_weights,name = 'output_box_outside_weights')
    # output_pred_class_one_batch=tf.convert_to_tensor(output_pred_class_one_batch,name= 'output_pred_class_one_batch')
    # output_labels              =tf.convert_to_tensor(output_labels,              name= 'output_labels')


    NUM_regression = cfg.TRAIN.NUM_REGRESSION
    NUM_class      = cfg.TRAIN.NUM_CLASSES
    RPN_BATCHSIZE  = cfg.TRAIN.RPN_BATCHSIZE

    pred_class = tf.reshape(pred_class, [batch_size, -1, NUM_class])
    pred_box   = tf.reshape(pred_box,   [batch_size, -1, NUM_regression])

    NUM_all_point = pred_class.get_shape()[1].value  ## all_point x num_anchor

    # output_pred_box_one_batch.set_shape([batch_size, ])

    # output_pred_box_one_batch.set_shape([batch_size, RPN_BATCHSIZE, NUM_regression])
    output_box_targets.set_shape([batch_size, RPN_BATCHSIZE, NUM_regression])
    output_box_inside_weights.set_shape([batch_size, RPN_BATCHSIZE, NUM_regression])
    output_box_outside_weights.set_shape([batch_size, RPN_BATCHSIZE, NUM_regression])
    # output_pred_class_one_batch.set_shape([batch_size, RPN_BATCHSIZE, NUM_class ])
    output_gt_class.set_shape([batch_size, RPN_BATCHSIZE])
    output_labels.set_shape([batch_size, NUM_all_point])

    indices_labels = tf.where(tf.greater_equal(output_labels,0))
    pred_class = tf.gather_nd(pred_class, indices_labels)
    pred_class = tf.reshape(pred_class, [batch_size,-1, NUM_class]) # batch: batch_size x num x num_class

    pred_box   = tf.gather_nd(pred_box, indices_labels)
    pred_box   = tf.reshape(pred_box, [batch_size, -1, NUM_regression])   # batch: batch_size x num x num_regression
    # output_pred_box_one_batch  = tf.reshape(output_pred_box_one_batch,  [-1, NUM_regression])
    # output_box_targets         = tf.reshape(output_box_targets ,        [-1, NUM_regression])
    # output_box_inside_weights  = tf.reshape(output_box_inside_weights,  [-1, NUM_regression])
    # output_box_outside_weights = tf.reshape(output_box_outside_weights, [-1, NUM_regression])
    # output_pred_class_one_batch= tf.reshape(output_pred_class_one_batch,[-1, NUM_class])
    # output_labels              = tf.reshape(output_labels,              [-1, 1] )

    classification_loss = tf.losses.sparse_softmax_cross_entropy(labels = output_gt_class, logits= pred_class, weights=1.0)
    # classification_loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.to_int32(pred_class[:,:,0:1]), logits= pred_class, weights=1.0) # just for test
    regression_loss = _smooth_l1(cfg.TRAIN.SIGMA, pred_box, output_box_targets, output_box_inside_weights, output_box_outside_weights )
    all_loss = tf.add(classification_loss, tf.multiply(cfg.TRAIN.LAMBDA, regression_loss))


    tf.summary.scalar('classification loss', classification_loss)
    tf.summary.scalar('regression loss', regression_loss)

    return all_loss


def region_proposal_loss(pred_class, pred_box, gt_box, smpw, xyz):
    """
    Input: pred_class: Batch x Num_point x Num_anchors x Num_class
           pred_box: Batch x Num_point x NUm_acnhors x 7
           gt_box: Batch x Num_box x ( num_regression + 1 ) # 1 is the category
           xyz: Batch x Num_point x 3
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
    CLASS = cfg.TRAIN.NUM_CLASSES
    N = xyz.shape[1]
    CC = xyz.shape[2]
    NUM_regression = cfg.TRAIN.NUM_REGRESSION
    pred_class = pred_class.reshape(NUM_BATCH, -1, A, CLASS)
    pred_box  = pred_box.reshape(NUM_BATCH, -1, A, NUM_regression)

    rpn_batchsize = cfg.TRAIN.RPN_BATCHSIZE
    # output_pred_box_one_batch   = np.zeros((NUM_BATCH, rpn_batchsize, NUM_regression), dtype = np.float32)
    output_box_targets          = np.zeros((NUM_BATCH, rpn_batchsize, NUM_regression), dtype = np.float32)
    output_box_inside_weights   = np.zeros((NUM_BATCH, rpn_batchsize, NUM_regression), dtype = np.float32)
    output_box_outside_weights  = np.zeros((NUM_BATCH, rpn_batchsize, NUM_regression), dtype = np.float32)
    # output_pred_class_one_batch = np.zeros((NUM_BATCH, rpn_batchsize, CLASS), dtype = np.float32)
    output_gt_class             = np.zeros((NUM_BATCH, rpn_batchsize), dtype = np.int32)
    output_labels               = np.zeros((NUM_BATCH, N*A),dtype = np.int32)

    for n in range(0,NUM_BATCH):
        # N is the points number of xyz
        # all_alpha is the [N A] A anchors, reshape to [N*A 1]
        # dif_alpha is the angle substract with label
        # calculate the angle gap to select one anchor
        #N  =  xyz[n,:,:].shape[0]
        #CC =  xyz[n,:,:].shape[1]

        # estimate the central points distance, using broadcasting ops
        distance = euclidean_distances(xyz[n,:,:], gt_box[n,:,5:8])

        distance = np.tile(distance, (1,A))
        distance = distance.reshape(-1,  gt_box[n,:,:].shape[0]) # label[n].shape(0)	is the number of ground truth

        labels = np.zeros(shape=(N*A,1))
        labels.fill(-1)

        # decide positive and negative labels
        argmin_dist = distance.argmin(axis=1)
        min_dist    = distance[np.arange(distance.shape[0]), argmin_dist]
        gt_argmin_dist = distance.argmin(axis=0)
        gt_min_dist  = distance[gt_argmin_dist, np.arange(distance.shape[1])]
        gt_argmin_dist = np.where(distance == gt_min_dist)[0]

        all_alpha = np.tile(cfg.TRAIN.Alpha, (N,1))
        all_alpha = all_alpha.reshape(-1,1)
        dif_alpha = all_alpha - gt_box[n, argmin_dist, 4].reshape(-1,1)

        arg_alpha = np.where(np.absolute(dif_alpha[:,0]) <= cfg.TRAIN.POSITIVE_ALPHA)[0]
        assert arg_alpha.shape[0] == (dif_alpha.shape[0]/2)
        min_dist_negative_inds = np.where(min_dist > cfg.TRAIN.NEGATIVE_CEN_DIST)[0]
        labels[np.intersect1d(arg_alpha, min_dist_negative_inds)] = 0


        labels[np.intersect1d(arg_alpha, gt_argmin_dist)] = 1
        min_dist_positive_inds = np.where(min_dist < cfg.TRAIN.POSITIVE_CEN_DIST)[0]
        labels[np.intersect1d(arg_alpha, min_dist_positive_inds)] = 1

        # randomly sampling cfg.TRAIN.RPN_BATCHSIZE
        num_positive_labels = int( cfg.TRAIN.RPN_FG_FRACTION*cfg.TRAIN.RPN_BATCHSIZE )
        positive_inds = np.where(labels == 1)[0]
        if len(positive_inds) > num_positive_labels:
            disable_inds = npr.choice(\
                                      positive_inds, size = (len(positive_inds) - num_positive_labels), replace = False )
            labels[disable_inds] = -1

        num_negative_labels = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        negative_inds = np.where(labels == 0)[0]
        if len(negative_inds) > num_negative_labels:
            disable_inds = npr.choice(\
                                      negative_inds, size = (len(negative_inds) - num_negative_labels), replace = False )
            labels[disable_inds] = -1


        # caculat the box regression
        box_targets = np.zeros((N*A, NUM_regression), dtype = np.float32 )
        anch_boxes = np.zeros((N*A, NUM_regression), dtype = np.float32 )
        box_inside_weights = np.zeros((N*A, NUM_regression), dtype = np.float32 )
        box_outside_weights = np.zeros((N*A, NUM_regression), dtype = np.float32 )
        anch_boxes[:,0] = cfg.TRAIN.Anchors[0]
        anch_boxes[:,1] = cfg.TRAIN.Anchors[1]
        anch_boxes[:,2] = cfg.TRAIN.Anchors[2]
        anch_boxes[:,3] = all_alpha[:, 0]
        all_xyz         = np.tile(xyz[n,:,:],(1, A))
        all_xyz         = all_xyz.reshape(-1, CC)
        anch_boxes[:,4] = all_xyz[:,0]
        anch_boxes[:,5] = all_xyz[:,1]
        anch_boxes[:,6] = all_xyz[:,2]


        # outside weights and inside_weigths
        box_inside_weights[ labels[:,0] ==1,:] = np.array(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)
        box_outside_weights[labels[:,0] ==1,:] = np.ones((1, NUM_regression))*1.0/ cfg.TRAIN.NUM_REGRESSION # cfg.TRAIN.RPN_BATCHSIZE # modify this parameters
        box_outside_weights[labels[:,0] ==0,:] = np.ones((1, NUM_regression))*1.0/ cfg.TRAIN.RPN_BATCHSIZE

        # calculate the box targets between anchors and ground truth
        assert anch_boxes.shape[1] == cfg.TRAIN.NUM_REGRESSION
        # assert gt_box.shape[1] == cfg.TRAIN.NUM_REGRESSION
        box_targets   = bbox_transform(anch_boxes, gt_box[n, argmin_dist, 1:8])

        # classification loss
        # pred_class = tf.reshape(pred_class, [-1,2])
        labels = labels.reshape(-1)

        if ISDEBUG:
            print('batch_num:{}, arg_alpha:{}, dif_alpha:{}'.format(labels[labels>=0].shape[0], arg_alpha.shape, dif_alpha.shape ))
            # print('arg_alpha:{}'.format(arg_alpha.shape))
            # print('')

        # output_pred_box_one_batch[n,:,:]  = pred_box_one_batch[labels>=0,:]
        if (labels[labels>=0].shape[0]) == rpn_batchsize:
            output_box_targets[n,:,:]         = box_targets[labels>=0,:]
            output_box_inside_weights[n,:,:]  = box_inside_weights[labels>=0,:]
            output_box_outside_weights[n,:,:] = box_outside_weights[labels>=0,:]
            output_labels[n,:]   = labels  # shape: batch x all_points
            output_gt_class[n,:] = labels[ labels>=0 ]
        else:
            output_box_targets[n,:,:]         = np.resize( box_targets[labels>=0,:], [rpn_batchsize, NUM_regression] )
            output_box_inside_weights[n,:,:]  = np.resize( box_inside_weights[labels>=0,:], [rpn_batchsize, NUM_regression] )
            output_box_outside_weights[n,:,:] = np.resize( box_outside_weights[labels>=0,:], [rpn_batchsize, NUM_regression] )
            output_labels[n,:]   = labels  # shape: batch x all_points
            output_gt_class[n,:] = np.resize(labels[labels>=0], [rpn_batchsize])
            if ISDEBUG:
                print('padding the laels:{}'.format(output_box_targets[n,:,:].shape))



    return output_box_targets, output_box_inside_weights, output_box_outside_weights, output_gt_class, output_labels


def _smooth_l1(sigma ,box_pred, box_targets, box_inside_weights, box_outside_weights):
    """
        loss = bbox_outside_weights*smoothL1(inside_weights * (box_pred - box_targets))
        smoothL1(x) = 0.5*(sigma*x)^2, if |x| < 1 / sigma^2
                       |x| - 0.5 / sigma^2, otherwise
    """
    sigma2 = sigma * sigma
    inside_mul = tf.multiply(box_inside_weights, tf.subtract(box_pred, box_targets)) # shape: Batch x rpn_batchsize x num_regression
    # inside_mul = box_inside_weights*(box_pred - box_targets)

    # smooth_l1_sign    = tf.cast(tf.less(tf.abs(inside_mul), 1.0/sigma2), tf.float32)  # shape: batch x rpn_batchszie x num_regression

    smooth_l1_sign = tf.stop_gradient(tf.to_float(tf.less(tf.abs(inside_mul), 1. / sigma2)))
    # smooth_l1_sign =  (np.absolute(inside_mul) < (1.0/sigma2))*1
    smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul,inside_mul), 0.5*sigma2)
    # smooth_l1_option1 = (inside_mul)**2*sigma2*0.5
    smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5*sigma2)
    # smooth_l1_option2 = np.absolute(inside_mul) - 0.5/sigma2

    smooth_l1_result  = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                               tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))
    # smooth_l1_result =  smooth_l1_option1*smooth_l1_sign + smooth_l1_option2*np.absolute(smooth_l1_sign - 1)

    outside_mul   = tf.multiply(box_outside_weights, smooth_l1_result)
    # outside_mul =  box_outside_weights*smooth_l1_result
    loss_reg = tf.reduce_mean(tf.reduce_sum(outside_mul, [1,2]))

    return loss_reg




if __name__=='__main__':
    #with tf.Graph().as_default():
    #    inputs = tf.zeros((32,2048,3))
    #    num_class = cfg.TRAIN.NUM_CLASS
    #    net, _ , _ , _ = get_model(inputs, tf.constant(True), num_class)
    #    print(net)
    '''
    Generating data to debug the code
    Input:
        pred_class: Batch x Num_point x Num_anchors x Num_class
        pred_box: Batch x Num_point x NUm_acnhors x 7
        gt_box: Batch x Num_box x (regression+1)
        xyz: Batch x Num_point x 3
        smpw: Batch x Num_point
    '''
    Batch = 32
    Num_point = 8192
    Num_anchor = cfg.TRAIN.NUM_ANCHORS
    Num_class = cfg.TRAIN.NUM_CLASSES
    Num_box = 16
    Num_regression = cfg.TRAIN.NUM_REGRESSION
    Num_channel = 3
    pred_class = np.random.rand(Batch, Num_point, Num_anchor, Num_class)
    pred_box = np.random.rand(Batch, Num_point, Num_anchor, Num_regression)
    gt_box = np.random.randint(1, 100, size = ( Batch, Num_box, Num_regression + 1 ), dtype = np.int)*1.0
    gt_box[:,:,4] = np.random.rand(Batch, Num_box)*np.pi
    xyz = np.random.randint(1, 100, size=(Batch, Num_point, Num_channel), dtype = np.int)*1.0
    smpw = np.random.rand(Batch, Num_point)
    output_pred_box_one_batch, output_box_targets, output_box_inside_weights, output_box_outside_weights, output_pred_class_one_batch, output_labels \
        = region_proposal_loss(pred_class, pred_box, gt_box, smpw, xyz)
    print(all_loss)
