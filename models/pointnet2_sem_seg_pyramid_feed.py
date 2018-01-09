import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_fp_module

def placeholder_inputs(batch_size, block_sample,data_num_ele,label_num_ele,bidmap_shapes):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size,)+ block_sample+ (data_num_ele,))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size,)+ block_sample+(label_num_ele,))
    smpws_pl = tf.placeholder(tf.float32, shape=(batch_size,)+ block_sample+(label_num_ele,))
    bidmap_pls = []
    for i in range(len(bidmap_shapes)):
        bidmap_pls.append( tf.placeholder(tf.int32,shape=(bidmap_shapes[i])) )
    return pointclouds_pl, labels_pl, smpws_pl, bidmap_pls


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

def get_model(grouped_rawdata, is_training, num_class, bidmaps, bidmaps_inverse, bn_decay=None):
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
    l_xyz[0] = grouped_rawdata
    l_points[0] = None
    bidmaps = [None] + bidmaps
    bidmaps_inverse = bidmaps_inverse

    for k in range(cascade_num):
        new_xyz, new_points = pointnet_sa_module(k,l_xyz[k], l_points[k], bidmaps[k], mlps[k], mlp2s[k], is_training=is_training, bn_decay=bn_decay, scope='layer'+str(k))
        l_xyz.append(new_xyz)
        l_points.append(new_points)

    end_points['l0_points'] = l_points[0]

    # Feature Propagation layers
    mlps = get_fp_module_config()
    for i in range(cascade_num):
        k = cascade_num-1 - i
        l_points[k-1] = pointnet_fp_module(l_points[k-1], l_points[k], bidmaps_inverse[k], mlps[k], is_training, bn_decay, scope='fa_layer'+str(i))

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    end_points['feats'] = net
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, num_class, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net, end_points


def get_loss(pred, label, smpw):
    """ pred: BxNxC,
        label: BxN,
	smpw: BxN """
    #label_category = label[:,:,label_eles_idx['label_category'][0]]
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=smpw)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)
