import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../utils_xyz'))
sys.path.append(os.path.join(BASE_DIR, '../config'))

from config_3d_obj_detection import cfg
import tensorflow as tf
import numpy as np
import numpy.random as npr
import tf_util
#from pointnet_blockid_sg_util_kitti import pointnet_sa_module, pointnet_fp_module
from pointnet_blockid_sg_util import pointnet_sa_module, pointnet_fp_module   ## benz_m using the latest version
import copy

from bbox_transform_2d import bbox_transform_2d
from nms_2d import convert_to_list_points

#from soft_cross_entropy import softmaxloss
from shapely.geometry import box, Polygon


small_addon_for_BCE = cfg.SMALL_addon_for_BCE
alpha = cfg.ALPHA
beta  = cfg.BETA
sigma = cfg.SIGMA


ISDEBUG = False
TMPDEBUG = True
def get_flatten_bidxmap_concat( flatten_bidxmaps, flatten_bm_extract_idx, cascade_id ):
        '''
            flatten_bidxmaps: (2, 26368, 2)
            flatten_bm_extract_idx:
                array([[    0,     0],
                       [25600,     2],
                       [26112,     2],
                       [26368,     2]], dtype=int32)
        '''
        batch_size = flatten_bidxmaps.get_shape()[0].value
        start = flatten_bm_extract_idx[cascade_id]
        end = flatten_bm_extract_idx[cascade_id+1]
        flatten_bidxmap_i = flatten_bidxmaps[ :,start[0]:end[0],: ]

        batch_idx = tf.reshape( tf.range(batch_size),[batch_size,1,1] )
        flatten_bidxmap_i_shape1 = flatten_bidxmap_i.get_shape()[1].value
        batch_idx = tf.tile( batch_idx,[1,flatten_bidxmap_i_shape1,1] )
        flatten_bidxmap_i_concat = tf.concat( [batch_idx,flatten_bidxmap_i],axis=-1,name="flatten_bidxmap%d_concat"%(cascade_id) )
        return flatten_bidxmap_i_concat

def placeholder_inputs(batch_size, NUM_POINT, data_num_ele, num_rpn_points, sg_bidxmaps_shape, num_regression, num_anchors):  ## benz_m
    sg_bidxmaps_shape = sg_bidxmaps_shape
    # flatten_bidxmaps_shape = sgf_configs['flatten_bidxmaps_shape']   ##   benz_m
    # flatten_bm_extract_idx = sgf_configs['flatten_bm_extract_idx']
    sgf_config_pls = {}
    with tf.variable_scope("pls") as pl_sc:
        #pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size,)+ num_rpn_points + (data_num_ele,))
        pointclouds_pl = tf.placeholder(tf.float32, [batch_size, NUM_POINT, data_num_ele])
        # sg_bidxmaps_pl = tf.placeholder( tf.int32,shape= (batch_size,) + sg_bidxmaps_shape )
        sg_bidxmaps_pl = tf.placeholder( tf.int32,shape= [batch_size, sg_bidxmaps_shape[0]-1,22])
        targets = tf.placeholder( tf.float32, [ batch_size, num_rpn_points, num_regression*num_anchors])
        ## postive anchors equal to one and others equal to zero(2 anchors in 1 position)
        positive_equal_one = tf.placeholder( tf.float32, [batch_size, num_rpn_points, num_anchors])
        positive_equal_one_sum = tf.placeholder(tf.float32, [batch_size, 1,  1])
        positive_equal_one_for_regression =  tf.placeholder( tf.float32, [ batch_size, num_rpn_points, num_regression*num_anchors])
        # negative anchors equal to one and others equal to zero
        negative_equal_one = tf.placeholder(tf.float32, [ batch_size, num_rpn_points, num_anchors])
        negative_equal_one_sum = tf.placeholder(tf.float32, [batch_size, 1, 1])

        sgf_config_pls['globalb_bottom_center_xyz'] = tf.placeholder(tf.float32, shape=(batch_size,1,6),name="globalb_bottom_center_xyz")

        return pointclouds_pl, sg_bidxmaps_pl, targets, positive_equal_one, positive_equal_one_sum, positive_equal_one_for_regression, \
                                                        negative_equal_one, negative_equal_one_sum, sgf_config_pls


def get_sa_module_config(model_flag):
    if model_flag[1] == 'V':
        return get_voxel3dcnn_sa_config(model_flag)
    else:
        return get_pointmax_sa_config(model_flag)

def get_voxel3dcnn_sa_config( model_flag ):
    cascade_num = int(model_flag[0])
    mlp_pe = []
    mlp_be = []
    voxel_channels = []
    voxel_kernels = []
    voxel_strides = []
    if model_flag=='4Va':
        voxel_channels.append( [32,32,64] )
        voxel_channels.append( [64,64,128] )
        voxel_channels.append( [128,128,256] )
        voxel_channels.append( [256,256,512] )
        for l in range(4):
            #voxel_kernels.append(  [2, 2, 1 ]  )
            #voxel_strides.append(  [1, 1, 1 ]  )
            mlp_pe.append([])
            mlp_be.append([])
    elif model_flag=='5VaG':
        voxel_channels.append( [32,32,48] )
        voxel_channels.append( [48,48,64] )
        voxel_channels.append( [64,64,128] )
        voxel_channels.append( [128,128,256] )
        voxel_channels.append( [256,256,512,512] )
        for l in range(5):
            #voxel_kernels.append(  [2, 2, 1 ]  )
            #voxel_strides.append(  [1, 1, 1 ]  )
            mlp_pe.append([])
            mlp_be.append([])

    mlp_configs = {}
    mlp_configs['voxel_channels'] = voxel_channels
    #mlp_configs['voxel_kernels'] = voxel_kernels
    #mlp_configs['voxel_strides'] = voxel_strides
    mlp_configs['point_encoder'] = mlp_pe
    mlp_configs['block_learning'] = '3DCNN'
    mlp_configs['block_encoder'] = mlp_be
    return mlp_configs

def get_pointmax_sa_config(model_flag):
    cascade_num = int(model_flag[0])
    mlp_pe = []
    if model_flag=='1a' or model_flag=='1aG':
        #mlp_pe.append( [64,64,128,128,512,1024] )
        mlp_pe.append( [64,64,64,128,512] )
    elif model_flag=='1b' or model_flag=='1bG':
        mlp_pe.append( [32, 64,64,128,128,256,512] )
    elif model_flag=='2a' or model_flag=='2aG':
        mlp_pe.append( [32,64,64,128] )
        mlp_pe.append( [128,128,256,512] )
    elif model_flag=='3a' or model_flag=='3aG':
        mlp_pe.append( [32,32,64] )
        mlp_pe.append( [64,128,256] )
        mlp_pe.append( [256,256,512] )
    elif model_flag=='4a' or model_flag=='4aG':
        mlp_pe.append( [32,32,64] )
        mlp_pe.append( [64,64,128] )
        mlp_pe.append( [128,128,256] )
        mlp_pe.append( [256,256,512] )
    elif model_flag=='4bG':
        mlp_pe.append( [24,24,48] )
        mlp_pe.append( [48,48,64] )
        mlp_pe.append( [64,64,128] )
        mlp_pe.append( [128,128,256] )
    elif model_flag=='5aG':
        mlp_pe.append( [32,32,64] )
        mlp_pe.append( [64,64,128] )
        mlp_pe.append( [128,128,256] )
        mlp_pe.append( [256,256,256] )
        mlp_pe.append( [256,512,512] )
    elif model_flag=='5bG':
        mlp_pe.append( [32,32,48] )
        mlp_pe.append( [48,48,64] )
        mlp_pe.append( [64,64,128] )
        mlp_pe.append( [128,128,256] )
        mlp_pe.append( [256,512,512] )

    elif model_flag=='1DSa' or model_flag=='1DSaG':
        dense_config = {}
        dense_config['num_block'] = 1
        dense_config['growth_rate'] = 32
        dense_config['initial_feature_num']=int(dense_config['growth_rate']*1.5)
        dense_config['layers_per_block'] = 5
        dense_config['keep_prob'] = 0.3
        mlp_pe.append( dense_config )

    elif model_flag=='4DSa' or model_flag=='4DSaG':
        dense_config = {}
        dense_config['num_block'] = 1
        dense_config['growth_rate'] = 16
        #dense_config['initial_feature_num']=int(dense_config['growth_rate']*1)
        dense_config['layers_per_block'] = 4
        dense_config['keep_prob'] = 0.3
        mlp_pe.append( copy.deepcopy(dense_config) )

        #del dense_config['initial_feature_num']
        #dense_config['initial_feature_rate'] = 0.8
        dense_config['layers_per_block'] = 3
        mlp_pe.append( dense_config )
        mlp_pe.append( dense_config )
        mlp_pe.append( dense_config )
    else:
        assert False,"model_flag not recognized: %s"%(model_flag)

    mlp_be = []
    if model_flag=='1a' or model_flag=='1aG':
        mlp_be.append( [512,256,128] )
    elif model_flag=='1b' or model_flag=='1bG':
        mlp_be.append( [512, 512, 512] )
    elif model_flag=='1DSa' or model_flag=='1DSaG':
        dense_config = {}
        dense_config['num_block'] = 1
        dense_config['growth_rate'] = 32
        dense_config['layers_per_block'] = 2
        dense_config['transition_feature_rate'] = 1
        dense_config['keep_prob'] = 0.3
        mlp_be.append( dense_config )
    else:
        for k in range(cascade_num):
            mlp_be.append( [] )

    mlp_configs = {}
    mlp_configs['point_encoder'] = mlp_pe
    mlp_configs['block_learning'] = 'max'
    mlp_configs['block_encoder'] = mlp_be
    return mlp_configs



def get_fp_module_config( model_flag ):
    cascade_num = int(model_flag[0])
    mlps_e1 = []
    if model_flag=='1b' or model_flag=='1bG':
        #mlps_e1.append( [128] )
        mlps_e1.append( [] )
    else:
        for k in range(cascade_num):
            mlps_e1.append( [] )

    mlps_fp = []
    if model_flag=='1a' or model_flag=='1aG' or model_flag=='1DSa' or model_flag=='1DSaG':
        mlps_fp.append( [512,256,128] )
    elif model_flag=='1b' or model_flag=='1bG':
        mlps_fp.append( [512, 256,  128, 128, 128] )
    elif model_flag=='2a' or model_flag=='2aG':
        mlps_fp.append( [256,128,128] )
        mlps_fp.append( [512,256,256] )
    elif model_flag=='3a' or model_flag=='3aG':
        mlps_fp.append( [128,128,128] )
        mlps_fp.append( [256,128] )
        mlps_fp.append( [256,256] )
    elif model_flag=='4aG' or model_flag=='4DSaG':
        mlps_fp.append( [128,128,128] )
        mlps_fp.append( [256,128] )
        mlps_fp.append( [256,256] )
        mlps_fp.append( [384,256] ) # for l_points[3-4]
    elif model_flag=='4bG':
        mlps_fp.append( [128,64,64] )
        mlps_fp.append( [128,128] )
        mlps_fp.append( [256,128] )
        mlps_fp.append( [384,256] ) # for l_points[3-4]
    elif model_flag=='5aG':
        mlps_fp.append( [128,128,128] )
        mlps_fp.append( [256,128] )
        mlps_fp.append( [256,256] )
        mlps_fp.append( [256,256] )
        mlps_fp.append( [512,256] )
    elif model_flag=='5bG':
        mlps_fp.append( [64,64,32] )
        mlps_fp.append( [128,64] )
        mlps_fp.append( [128,128] )
        mlps_fp.append( [256,128] )
        mlps_fp.append( [512,256] )
    #elif model_flag=='1DSa' or model_flag=='1DSaG':
    #    dense_config = {}
    #    dense_config['num_block'] = 1
    #    dense_config['growth_rate'] = 24
    #    dense_config['layers_per_block'] = 4
    #    dense_config['transition_feature_rate'] = 1
    #    dense_config['keep_prob'] = 0.5
    #    mlps_fp.append( dense_config )
    else:
        assert False, "model flag %s not exist"%(model_flag)

    return mlps_e1, mlps_fp

def shape_str(tensor_ls):
    shape_str = ''
    for i in range(len(tensor_ls)):
        if tensor_ls[i] == None:
            shape_str += '\t None'
        else:
            shape_str += '\t' + str( [s.value for s in tensor_ls[i].shape] )
        if i < len(tensor_ls)-1:
            shape_str += '\n'
    return shape_str

def get_global_bidxmap( batch_size, nsubblock_last ):
    sg_bidxmap_global = tf.reshape( tf.range(nsubblock_last, dtype=tf.int32),[1,1,nsubblock_last] )
    sg_bidxmap_global = tf.tile( sg_bidxmap_global,[batch_size,1,1] )
    return sg_bidxmap_global

def get_flatten_bidxmap_global( batch_size, nsubblock_last, nearest_block_num ):
    '''
       flatten_bidxmap: (B,256,self.flatbxmap_max_nearest_num,3)
                      N: base_bid_index
                     [:,:,0]: aim_b_index
                     [:,:,1]: point_index_in_aimb  (useless when cascade_id>0)
                     [:,:,2]: index_distance
    '''
    #tmp = tf.constant(value=[0], shape = [1,1,1,1], dtype=tf.int32)
    #tmp = tf.tile( tmp,[1, nsubblock_last, 1, 1] )
    tmp = tf.zeros( shape=[1, nsubblock_last, 1, 1], dtype=tf.int32 )
    flatten_bidxmap_global = tf.reshape( tf.range( nsubblock_last ),(1,-1,1,1) )
    flatten_bidxmap_global = tf.concat( [tmp, flatten_bidxmap_global, tmp], -1 )
    flatten_bidxmap_global = tf.tile( flatten_bidxmap_global,[batch_size, 1, nearest_block_num, 1] )

    fbmap_neighbor_dis_global = tf.zeros(shape = [batch_size, nsubblock_last, nearest_block_num, 1], dtype=tf.float32)
    return flatten_bidxmap_global, fbmap_neighbor_dis_global

#def get_model(modelf_nein, rawdata, is_training, num_class, sg_bidxmaps, flatten_bidxmaps, fbmap_neighbor_dis, sgf_configs, bn_decay=None, IsDebug=False):

def get_model(modelf_nein, rawdata, is_training, num_class, num_anchors, num_regression, sg_bidxmaps, sgf_configs, sgf_config_pls, bn_decay=None,IsDebug = False):     ## benz_m
    """
        rawdata: (B, global_num_point, 6)   (xyz is at first 3 channels)
        out: (N,n1,n2,class)
    """
    IsShowModel = True
    model_flag, num_neighbors = modelf_nein.split('_')
    if 'G' in model_flag:
        IsAddGlobalLayer = True
    else:
        IsAddGlobalLayer = False

    flatten_bm_extract_idx = sgf_configs['flatten_bm_extract_idx']
    sg_bm_extract_idx = sgf_configs['sg_bm_extract_idx']

    batch_size = rawdata.get_shape()[0].value
    global_num_point = rawdata.get_shape()[1].value
    end_points = {}

    cascade_num = int(model_flag[0])
    assert cascade_num <= sg_bm_extract_idx.shape[0]+(1*IsAddGlobalLayer)  # sg_bm_extract_idx do not include the global step
    mlp_configs = get_sa_module_config(model_flag)
    l_points = []                       # size = l_points+1
    l_points.append( rawdata )
    l_xyz = rawdata[...,0:3]     # (2, 512, 128, 6)
    debug = {}
    if IsDebug:
        debug['l_xyz'] = []
        debug['l_xyz'].append( l_xyz )
        debug['grouped_xyz'] = []
        debug['flat_xyz'] = []

    if IsShowModel: print('\n\ncascade_num:%d \ngrouped_rawdata:%s'%(cascade_num, shape_str([rawdata]) ))
    sgf_config_pls['max_step_stride'] = (sgf_config_pls['globalb_bottom_center_xyz'][:,:,3:6] - sgf_config_pls['globalb_bottom_center_xyz'][:,:,0:3]) * tf.constant(2,tf.float32)

    for k in range(cascade_num):
        IsExtraGlobalLayer = False
        start = sg_bm_extract_idx[k]
        end = sg_bm_extract_idx[k+1]
        sg_bidxmap_k = sg_bidxmaps[ :,start[0]:end[0],0:end[1] ]
        block_bottom_center_mm = sg_bidxmaps[ :,start[0]:end[0],end[1]:end[1]+6 ]

        #if TMPDEBUG:
        #    pooling = 'max' # 3DCNN

        #l_xyz, new_points, root_point_features, grouped_xyz = pointnet_sa_module(k, IsExtraGlobalLayer, l_xyz, l_points[k], sg_bidxmap_k,  mlps_0[k], mlps_1[k], block_center_xyz_mm, sgf_configs,
        #                                                            is_training=is_training, bn_decay=bn_decay, pooling=pooling, scope='sa_layer'+str(k) )
        l_xyz, new_points, root_point_features = pointnet_sa_module(k,  l_xyz, l_points[k], sg_bidxmap_k,  mlp_configs, block_bottom_center_mm,\
                                                                                 sgf_configs, sgf_config_pls, is_training=is_training, bn_decay=bn_decay, scope='sa_layer'+str(k) )

        #if IsDebug:
        #    debug['l_xyz'].append( l_xyz )
        #    debug['grouped_xyz'].append( grouped_xyz )
        if k == 0:
            l_points[0] = root_point_features
        l_points.append(new_points)

        # l_xyz: (2, 512, 128, 6) (2, 512, 3)  (2, 256, 3) (2, 64, 3)
        # l_points: None  (2, 512, 64) (2, 256, 256) (2, 64, 512)
        if IsShowModel: print('pointnet_sa_module %d, l_xyz: %s'%(k,shape_str([l_xyz])))
    if IsShowModel: print('\nafter pointnet_sa_module, l_points:\n%s'%(shape_str(l_points)))
    end_points['l0_points'] = l_points[0]

    # Feature Propagation layers
    '''
     mlps_e1, mlps_fp = get_fp_module_config( model_flag )
    for i in range(cascade_num):
        k = cascade_num-1-i
        if IsAddGlobalLayer and k==cascade_num-1:
            if k==0: nsubblock_last = l_points[k].get_shape()[2].value # l_points[0] is grouped feature
            else: nsubblock_last = l_points[k].get_shape()[1].value  # l_points[0] is flat feature
            flatten_bidxmaps_k, fbmap_neighbor_dis_k = get_flatten_bidxmap_global( batch_size, nsubblock_last, flatten_bidxmaps.get_shape()[2].value )
        else:
            start = flatten_bm_extract_idx[k]
            end = flatten_bm_extract_idx[k+1]
            flatten_bidxmaps_k = flatten_bidxmaps[ :,start[0]:end[0],:,: ]
            fbmap_neighbor_dis_k =  fbmap_neighbor_dis[:,start[0]:end[0],:,:]
        l_points[k] = pointnet_fp_module( k, num_neighbors, l_points[k], l_points[k+1], flatten_bidxmaps_k, fbmap_neighbor_dis_k, mlps_e1[k],  mlps_fp[k], is_training, bn_decay, scope='fp_layer'+str(i), debug=debug )
    # l_points: (2, 25600, 128) (2, 512, 128) (2, 256, 256) (2, 64, 512)
    if IsShowModel: print('\nafter pointnet_fp_module, l_points:\n%s\n'%(shape_str(l_points)))

    # FC layers
    net = tf_util.conv1d(l_points[0], l_points[0].get_shape()[-1], 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    if IsShowModel: print('net:%s'%(shape_str([net])))
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    if IsShowModel: print('net:%s'%(shape_str([net])))
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, scope='fc2')
    '''
    net = tf_util.conv1d(new_points, 2048, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    if IsShowModel: print('net:%s'%(shape_str([net])))
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')

    ## num_class = 1, num_anchors = 2, num_regression = 6, ignore z
    net_class = tf_util.conv1d(net, num_anchors*num_class     , 1 , padding='VALID', activation_fn=None, scope='fc2') # outputing the classification for every point
    net_boxes = tf_util.conv1d(net, num_anchors*num_regression, 1 , padding='VALID', activation_fn=None, scope='fc3') # outputing the 3D bounding boxes

    return net_class, net_boxes, l_xyz ## benz_m, l_xyz should be center point of every box


def get_loss( BATCH_SIZE, pred_class_feature, pred_box_feature, xyz_pl, targets ,positive_equal_one, positive_equal_one_sum, positive_equal_one_for_regression, negative_equal_one, negative_equal_one_sum):
    '''
    the targets are precomputed based on IoU between anchor_boxes and ground_truth
    '''
    pred_probability = tf.sigmoid(pred_class_feature)   ## convert to probability, it is binary classification problem,  1/(1+e^(-x)) = e^x /(e^x + e^0), very similar with softmax

    classification_positive_loss = ( -positive_equal_one*tf.log( pred_probability + small_addon_for_BCE)) / positive_equal_one_sum
    classification_negative_loss = ( -negative_equal_one*tf.log( 1 - pred_probability + small_addon_for_BCE)) / negative_equal_one_sum

    classification_loss = tf.reduce_sum( alpha*classification_positive_loss + beta*classification_negative_loss )

    regression_loss = smooth_l1( pred_box_feature*positive_equal_one_for_regression,  targets*positive_equal_one_for_regression, sigma ) / positive_equal_one_sum

    output_regression_loss = tf.reduce_sum(regression_loss)
    all_loss = tf.reduce_sum( classification_loss + output_regression_loss )

    output_classification_positive_loss = tf.reduce_sum(classification_positive_loss)
    output_classification_negative_loss = tf.reduce_sum(classification_negative_loss)


    #TODO: adding the recall and accuracy record later
    # recall =
    # accuracy =

    return all_loss, classification_loss, output_regression_loss, output_classification_positive_loss, output_classification_negative_loss


def smooth_l1(deltas, targets, sigma=3.0):
    """
        loss = bbox_outside_weights*smoothL1(inside_weights * (box_pred - box_targets))
        smoothL1(x) = 0.5*(sigma*x)^2, if |x| < 1 / sigma^2
                       |x| - 0.5 / sigma^2, otherwise
    """
    sigma2 = sigma * sigma
    diffs = tf.subtract(deltas, targets)
    smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)
    smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + \
        tf.multiply(smooth_l1_option2, 1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add

    return smooth_l1


def get_loss_old_birdview_los(batch_size, pred_class, pred_box, gt_box, xyz):
    '''
    pred_class: batch * num_point * num_anchor * 2
    pred_box  : batch * num_point * num_anchor * 7
    gt_box    : batch * n * (num_regression + 1) # 1 is the categroy
    xyz       : batch * num_point * 3
    '''
    gt_box = tf.convert_to_tensor(gt_box, tf.float32)
    output_box_targets, output_box_inside_weights, output_box_outside_weights, output_gt_class, output_labels, gt_class_recall = \
             tf.py_func(region_proposal_loss, [pred_class, pred_box, gt_box, xyz], [tf.float32, tf.float32, tf.float32, tf.int32, tf.int32, tf.int32])

    NUM_regression = cfg.TRAIN.NUM_REGRESSION
    NUM_class      = cfg.TRAIN.NUM_CLASSES
    RPN_BATCHSIZE  = cfg.TRAIN.RPN_BATCHSIZE
    NUM_anchors    = cfg.TRAIN.NUM_ANCHORS

    pred_class = tf.reshape(pred_class, [batch_size, -1, NUM_class])
    pred_box   = tf.reshape(pred_box,   [batch_size, -1, NUM_regression])

    pred_prob = tf.nn.softmax(pred_class)
    pred_prob = tf.reshape(pred_prob, [batch_size, -1, NUM_class*NUM_anchors])

    NUM_all_point = pred_class.get_shape()[1].value  ## all_point x num_anchor

    output_box_targets.set_shape([batch_size, RPN_BATCHSIZE, NUM_regression])
    output_box_inside_weights.set_shape([batch_size, RPN_BATCHSIZE, NUM_regression])
    output_box_outside_weights.set_shape([batch_size, RPN_BATCHSIZE, NUM_regression])
    # output_pred_class_one_batch.set_shape([batch_size, RPN_BATCHSIZE, NUM_class ])
    output_gt_class.set_shape([batch_size, RPN_BATCHSIZE])
    output_labels.set_shape([batch_size, NUM_all_point])
    gt_class_recall.set_shape([batch_size, RPN_BATCHSIZE])


    indices_labels = tf.where(tf.greater_equal(output_labels,0))
    pred_class = tf.gather_nd(pred_class, indices_labels)
    #print(pred_class.get_shape())
    pred_class = tf.reshape(pred_class, [batch_size,-1, NUM_class]) # batch: batch_size x num x num_class
    #tf.summary.scalar('pred_class_1', tf.shape(pred_class))

    pred_box   = tf.gather_nd(pred_box, indices_labels)
    #tf.summary.scalar('pred_box_0', tf.shape(pred_box))
    pred_box   = tf.reshape(pred_box, [batch_size, -1, NUM_regression])   # batch: batch_size x num x num_regression
    # output_box_targets         = tf.reshape(output_box_targets ,        [-1, NUM_regression])

    classification_loss = tf.losses.sparse_softmax_cross_entropy(labels = output_gt_class, logits= pred_class, weights=1.0)
    # classification_loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.to_int32(pred_class[:,:,0:1]), logits= pred_class, weights=1.0) # just for test
    regression_loss, loss_details = _smooth_l1_old(cfg.TRAIN.SIGMA, pred_box, output_box_targets, output_box_inside_weights, output_box_outside_weights )
    all_loss = tf.add(classification_loss, tf.multiply(cfg.TRAIN.LAMBDA, regression_loss))


    correct0 = tf.equal(tf.argmax(pred_class, 2), tf.to_int64(output_gt_class))
    accuracy_classification = tf.reduce_sum(tf.cast(correct0, tf.float32)) / float(batch_size*RPN_BATCHSIZE)

    num_recall = tf.equal(tf.to_int64(gt_class_recall), 1)
    correct1 = tf.equal(tf.argmax(pred_class, 2), tf.to_int64(gt_class_recall))
    recall_classification  = tf.reduce_sum(tf.cast(correct1, tf.float32)) / tf.reduce_sum(tf.cast(num_recall, tf.float32))

    num_positive_label     = tf.reduce_sum(tf.cast(num_recall, tf.float32))

    tf.summary.scalar('accuracy', accuracy_classification)
    tf.summary.scalar('recall', recall_classification)
    tf.summary.scalar('num_positive_label', num_positive_label)
    tf.summary.scalar('classification loss', classification_loss)
    tf.summary.scalar('regression loss', regression_loss)
    #tf.summary.tensor_summary('loss_details',loss_details)
    tf.summary.scalar('dl', loss_details[0])
    tf.summary.scalar('dw', loss_details[1])
    tf.summary.scalar('dtheta', loss_details[2])
    tf.summary.scalar('dx', loss_details[3])
    tf.summary.scalar('dy', loss_details[4])
    return all_loss, classification_loss, regression_loss, loss_details, pred_prob, accuracy_classification, recall_classification, num_positive_label


def region_proposal_loss(pred_class, pred_box, gt_box, xyz):
    """
    Input: pred_class: Batch x Num_point x Num_anchors x Num_class
           pred_box: Batch x Num_point x NUm_acnhors x num_regression
           gt_box: Batch x Num_box x ( num_regression + 1 ) # 1 is the category
           xyz: Batch x Num_point x 3
    Description: similar to faster rcnn, overlaps between prediction boxes and ground truth boxes are estimated to decide point labels and further calucuate the box_targets
    --> preparing all anchors in every space point --> calculating overlaps bewteen anchor boxes and bounding boxes -->  anchors whose overlaps are over a certain threshold are regarded as positive labels, the rest are negative labels
    --> calculate the box_targets between anchors and ground truth --> prepare the outside_weight and inside_weight -->  then write the loss function
    """

    assert xyz.shape[2] == 3
    NUM_BATCH = xyz.shape[0]
    A = cfg.TRAIN.NUM_ANCHORS
    CLASS = cfg.TRAIN.NUM_CLASSES
    N = xyz.shape[1]
    CC = xyz.shape[2]
    NUM_regression = cfg.TRAIN.NUM_REGRESSION
    pred_class = pred_class.reshape(NUM_BATCH, -1, A, CLASS)
    pred_box  = pred_box.reshape(NUM_BATCH, -1, A, NUM_regression)
    anchor_coordinate_list = cfg.TRAIN.Anchor_coordinate_lists  # 2*5*2

    positive_samples_threshold = cfg.TRAIN.POSITIVE_THRESHOLD
    negative_samples_threshold = cfg.TRAIN.NEGATIVE_THRESHOLD

    rpn_batchsize = cfg.TRAIN.RPN_BATCHSIZE
    output_box_targets          = np.zeros((NUM_BATCH, rpn_batchsize, NUM_regression), dtype = np.float32)
    output_box_inside_weights   = np.zeros((NUM_BATCH, rpn_batchsize, NUM_regression), dtype = np.float32)
    output_box_outside_weights  = np.zeros((NUM_BATCH, rpn_batchsize, NUM_regression), dtype = np.float32)
    output_gt_class             = np.zeros((NUM_BATCH, rpn_batchsize), dtype = np.int32)
    gt_class_recall             = np.zeros((NUM_BATCH, rpn_batchsize), dtype = np.int32)
    output_labels               = np.zeros((NUM_BATCH, N*A),dtype = np.int32)


    min_infinite =  np.finfo(np.float64).eps

    for n in range(0,NUM_BATCH):
        # step1, inverse transformation to getting all anchor bounding boxes
        # step2, compare gt_boxes with anchor boxes, getting the overlapping
        # matrix similiar to distance matrix.

        # getting all coordinates for the regions
        point_xy = xyz[n,:,0:2].reshape(-1,1,1,2)
        all_anchor_coordinate_list = point_xy + anchor_coordinate_list           ## shape: n*2*5*2
        all_anchor_coordinate_list = all_anchor_coordinate_list.reshape(-1,5,2)

        all_alpha = np.tile(cfg.TRAIN.Alpha, ( N, 1))
        all_alpha = all_alpha.reshape(-1,1)
        #dif_alpha = all_alpha - gt_box[n, argmin_dist, 4].reshape(-1,1)

        if False:
            print('----------batch:{}----------'.format(n))
            print('gt shape is {}'.format(gt_box[n,:,:].shape))


        N_anchors    =  all_anchor_coordinate_list.shape[0]
        all_anchor_coordinate_list = all_anchor_coordinate_list.tolist()
        ## preparing all the polygon of anchors to save computation time
        anchor_polygon = []
        for i in range(N_anchors):
            anchor_polygon.append(Polygon( all_anchor_coordinate_list[i] ))


        gt_box_coordinate_list =  convert_to_list_points(gt_box[n, :, 1:6])
        N_gt_boxes = gt_box.shape[1]
        ## preparing all the polygon of gt boxes to save computation time
        gt_polygon = []
        for j in range(N_gt_boxes):
            gt_polygon.append(Polygon( gt_box_coordinate_list[j]))

        # estimating the overlap between differnt ploygons

        overlap_mat = np.array([[ anchor_polygon[i].intersection( gt_polygon[j] ).area  for j in range(N_gt_boxes)] for i in range(N_anchors)])
        overlap_mat = np.array([[ overlap_mat[i, j]/np.maximum(  anchor_polygon[i].area + gt_polygon[j].area - overlap_mat[i,j], min_infinite)  for j in range(N_gt_boxes)] for i in range(N_anchors)])

        labels = np.zeros(shape=(N_anchors, 1))
        labels.fill(-1)

        # Decide positive and negative arguments
        argmax_overlap = overlap_mat.argmax(axis = 1)
        max_overlap = overlap_mat[np.arange(overlap_mat.shape[0]), argmax_overlap]
        gt_argmax_overlap = overlap_mat.argmax(axis = 0)
        gt_max_overlap  =  overlap_mat[gt_argmax_overlap, np.arange(overlap_mat.shape[1])]

        gt_argmax_overlap = np.where(overlap_mat == gt_max_overlap)[0]


        labels[gt_argmax_overlap] = 1
        ## Decide the positive and negative samples
        arg_negative = np.where( max_overlap < negative_samples_threshold)[0]
        labels[arg_negative] = 0
        arg_positive = np.where( max_overlap > positive_samples_threshold)[0]
        labels[arg_positive] = 1
        # print('arg_negative:{}, arg_positive:{}, gt_argmax_overlap:{}'.format(arg_negative.shape, arg_positive.shape, gt_argmax_overlap.shape))

        # randomly sampling cfg.TRAIN.RPN_BATCHSIZE
        num_positive_labels = int( cfg.TRAIN.RPN_FG_FRACTION*cfg.TRAIN.RPN_BATCHSIZE )
        positive_inds = np.where(labels == 1)[0]
        if len(positive_inds) == 0:      ##in case that there is no positive labels, so just random choosing one, and setting it positive, to prevent bugging
            able_inds = npr.choice(\
                                   arg_negative,  size = 1, replace = False )
            labels[able_inds] = 1

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
        box_targets = np.zeros((N_anchors, NUM_regression), dtype = np.float32 )
        anch_boxes = np.zeros(( N_anchors, NUM_regression), dtype = np.float32 )
        box_inside_weights  = np.zeros((N_anchors, NUM_regression), dtype = np.float32 )
        box_outside_weights = np.zeros((N_anchors, NUM_regression), dtype = np.float32 )
        anch_boxes[:,0] = cfg.TRAIN.Anchors[0]
        anch_boxes[:,1] = cfg.TRAIN.Anchors[1]
        # anch_boxes[:,2] = cfg.TRAIN.Anchors[2]
        anch_boxes[:,2] = all_alpha[:, 0]
        all_xyz         = np.tile(xyz[n,:,:],(1, A))
        all_xyz         = all_xyz.reshape(-1, CC)
        anch_boxes[:,3] = all_xyz[:,0]
        anch_boxes[:,4] = all_xyz[:,1]
        # anch_boxes[:,6] = all_xyz[:,2]


        # outside weights and inside_weigths
        box_inside_weights[ labels[:,0] ==1,:] = np.array(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)
        box_outside_weights[labels[:,0] ==1,:] = np.ones((1, NUM_regression))*1.0/ cfg.TRAIN.NUM_REGRESSION # cfg.TRAIN.RPN_BATCHSIZE # modify this parameters
        box_outside_weights[labels[:,0] ==0,:] = np.ones((1, NUM_regression))*1.0/ cfg.TRAIN.RPN_BATCHSIZE

        # calculate the box targets between anchors and ground truth
        assert anch_boxes.shape[1] == cfg.TRAIN.NUM_REGRESSION
        # assert gt_box.shape[1] == cfg.TRAIN.NUM_REGRESSION
        box_targets   = bbox_transform_2d(anch_boxes, gt_box[n, argmax_overlap, 1:6])

        # classification loss
        # pred_class = tf.reshape(pred_class, [-1,2])
        labels = labels.reshape(-1)

        # if ISDEBUG:
            # print('batch_num:{}, arg_alpha:{}, dif_alpha:{}'.format(labels[labels>=0].shape[0], arg_alpha.shape, dif_alpha.shape ))
            # print('arg_alpha:{}'.format(arg_alpha.shape))
            # print('')

        # output_pred_box_one_batch[n,:,:]  = pred_box_one_batch[labels>=0,:]
        # print('label shape:{}, 1 label is:{}, 0 label is:{}'.format(labels[labels>=0].shape, labels[labels==1].shape, labels[labels==0].shape))
        assert (labels[labels>=0].shape[0]) == rpn_batchsize
        # print('box_targets shape:{}'.format(box_targets[labels>=0,:].shape))
        output_box_targets[n,:,:]         = box_targets[labels>=0,:]
        output_box_inside_weights[n,:,:]  = box_inside_weights[labels>=0,:]
        output_box_outside_weights[n,:,:] = box_outside_weights[labels>=0,:]
        output_labels[n,:]   = labels  # shape: batch x all_points
        output_gt_class[n,:] = labels[ labels>=0 ]
        labels[labels==0] = 10
        gt_class_recall[n,:] = labels[ labels>=0 ]
        #else:
        #    output_box_targets[n,:,:]         = np.resize( box_targets[labels>=0,:], [rpn_batchsize, NUM_regression] )
        #    output_box_inside_weights[n,:,:]  = np.resize( box_inside_weights[labels>=0,:], [rpn_batchsize, NUM_regression] )
        #    output_box_outside_weights[n,:,:] = np.resize( box_outside_weights[labels>=0,:], [rpn_batchsize, NUM_regression] )
        #    output_labels[n,:]   = labels  # shape: batch x all_points
        #    output_gt_class[n,:] = np.resize(labels[labels>=0], [rpn_batchsize])
        #    if ISDEBUG:
        #        print('padding the laels:{}'.format(output_box_targets[n,:,:].shape))
        if False:
            print('Ground truth shape is {}'.format(gt_box[n,:,:].shape))
            print('Positive label size is {}'.format(labels[labels>=1].shape))

    if ISDEBUG:
        print('box target shape is {}, box inside shape is {}, box outside is {}, label shape is{}, gt box shape is {}'.format(output_box_targets.shape, output_box_inside_weights.shape,
                                                                                                                            output_box_outside_weights.shape, output_labels.shape, output_gt_class.shape))

    return output_box_targets, output_box_inside_weights, output_box_outside_weights, output_gt_class, output_labels, gt_class_recall


def _smooth_l1_old(sigma ,box_pred, box_targets, box_inside_weights, box_outside_weights):
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
    smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5/sigma2)
    # smooth_l1_option2 = np.absolute(inside_mul) - 0.5/sigma2

    smooth_l1_result  = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                               tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(1. ,smooth_l1_sign))))
    # smooth_l1_result =  smooth_l1_option1*smooth_l1_sign + smooth_l1_option2*np.absolute(smooth_l1_sign - 1)

    outside_mul   = tf.multiply(box_outside_weights, smooth_l1_result)
    # outside_mul =  box_outside_weights*smooth_l1_result
    loss_reg = tf.reduce_mean(tf.reduce_sum(outside_mul, [1,2]))

    loss_details = tf.reduce_mean(tf.reduce_sum(outside_mul,1),0)

    return loss_reg, loss_details








if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)
