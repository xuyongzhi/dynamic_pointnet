import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../utils_xyz'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_blockid_sg_util import pointnet_sa_module, pointnet_fp_module

def flatten_grouped_labels(grouped_labels, grouped_smpws, flatten_bidxmap0 ):
    batch_size = grouped_labels.get_shape()[0].value
    batch_idx = tf.reshape( tf.range(batch_size),[batch_size,1,1] )
    flatten_bidxmap0_shape1 = flatten_bidxmap0.get_shape()[1].value
    batch_idx = tf.tile( batch_idx,[1,flatten_bidxmap0_shape1,1] )
    flatten_bidxmap0_concat = tf.concat( [batch_idx,flatten_bidxmap0],axis=-1 )

    label = tf.gather_nd(grouped_labels,flatten_bidxmap0_concat)
    smpw = tf.gather_nd(grouped_smpws,flatten_bidxmap0_concat)
    return label, smpw

def placeholder_inputs(batch_size, block_sample,data_num_ele,label_num_ele, sg_bidxmaps_shape, flatten_bidxmaps_shape, flatten_bm_extract_idx):
    grouped_pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size,)+ block_sample+ (data_num_ele,))
    grouped_labels_pl = tf.placeholder(tf.int32, shape=(batch_size,)+ block_sample+(label_num_ele,))
    grouped_smpws_pl = tf.placeholder(tf.float32, shape=(batch_size,)+ block_sample+(label_num_ele,))
    sg_bidxmaps_pl = tf.placeholder( tf.int32,shape= (batch_size,)+sg_bidxmaps_shape )
    flatten_bidxmaps_pl = tf.placeholder(tf.int32,shape= (batch_size,)+flatten_bidxmaps_shape)

    start = flatten_bm_extract_idx[0]
    end = flatten_bm_extract_idx[1]
    flatten_bidxmaps_pl_0 = flatten_bidxmaps_pl[ :,start[0]:end[0],: ]

    labels_pl, smpws_pl = flatten_grouped_labels(grouped_labels_pl, grouped_smpws_pl, flatten_bidxmaps_pl_0)
    debug={}
    debug['flatten_bidxmaps_pl_0'] = flatten_bidxmaps_pl_0
    return grouped_pointclouds_pl, grouped_labels_pl, grouped_smpws_pl, sg_bidxmaps_pl, flatten_bidxmaps_pl, labels_pl, smpws_pl, debug

def get_sa_module_config():
    mlps = []
    mlps.append( [32,32,64] )
    mlps.append( [64,64,128] )
    mlps.append( [128,128,256] )
    mlps.append( [256,256,512] )

    mlp2s = []
    for k in mlps:
        mlp2s.append(None )
    return mlps,mlp2s
def get_fp_module_config():
    mlps = []
    mlps.append( [128,128,128] )
    mlps.append( [256,128] )
    mlps.append( [256,256] )
    mlps.append( [256,256] )

    return mlps

def get_model(grouped_rawdata, is_training, num_class, sg_bidxmaps, sg_bm_extract_idx, flatten_bidxmaps, flatten_bm_extract_idx, bn_decay=None):
    """
        grouped_rawdata: (B,n1,n2,c)   (xyz is at first 3 channels)
        out: (N,n1,n2,class)
    """
    batch_size = grouped_rawdata.get_shape()[0].value
    block_sample0 = grouped_rawdata.get_shape()[1].value
    block_sample1 = grouped_rawdata.get_shape()[2].value
    end_points = {}

    mlps,mlp2s = get_sa_module_config()
    cascade_num = len(mlps)
    l_xyz = []
    l_points = []
    l_xyz.append( grouped_rawdata )
    l_points.append( None )

    for k in range(cascade_num):
        if k==0:
            sg_bidxmap_k = None
        else:
            start = sg_bm_extract_idx[k-1]
            end = sg_bm_extract_idx[k]
            sg_bidxmap_k = sg_bidxmaps[ :,start[0]:end[0],0:end[1] ]
        new_xyz, new_points = pointnet_sa_module(k,l_xyz[k], l_points[k], sg_bidxmap_k, mlps[k], mlp2s[k], is_training=is_training, bn_decay=bn_decay, scope='layer'+str(k))
        l_xyz.append(new_xyz)
        l_points.append(new_points)

    end_points['l0_points'] = l_points[0]

    # Feature Propagation layers
    mlps = get_fp_module_config()
    for i in range(cascade_num):
        k = cascade_num-1 - i
        start = flatten_bm_extract_idx[k]
        end = flatten_bm_extract_idx[k+1]
        flatten_bidxmaps_k = flatten_bidxmaps[ :,start[0]:end[0],: ]
        l_points[k] = pointnet_fp_module(l_points[k], l_points[k+1], flatten_bidxmaps_k, mlps[k], is_training, bn_decay, scope='fa_layer'+str(i))

    # FC layers
    net = tf_util.conv1d(l_points[0], mlps[0][-1], 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net, end_points


def get_loss(pred, label, smpw, label_eles_idx ):
    """ pred: BxNxC,
        label: BxN,
	smpw: BxN """
    category_idx = label_eles_idx['label_category'][0]
    label_category = label[...,category_idx]
    smpw_category = smpw[...,category_idx]

    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label_category, logits=pred, weights=smpw_category)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)
