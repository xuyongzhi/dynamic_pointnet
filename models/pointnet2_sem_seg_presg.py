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
import copy

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

def placeholder_inputs(batch_size, block_sample,data_num_ele,label_num_ele, sgf_configs):
    sg_bidxmaps_shape = sgf_configs['sg_bidxmaps_shape']
    flatten_bidxmaps_shape = sgf_configs['flatten_bidxmaps_shape']
    flatten_bm_extract_idx = sgf_configs['flatten_bm_extract_idx']
    cascade_num = flatten_bm_extract_idx.shape[0]-1
    sgf_config_pls = {}
    with tf.variable_scope("pls") as pl_sc:
        pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size,)+ block_sample + (data_num_ele,))
        labels_pl = tf.placeholder(tf.int32, shape=(batch_size,)+ block_sample + (label_num_ele,))
        smpws_pl = tf.placeholder(tf.float32, shape=(batch_size,)+ block_sample + (label_num_ele,))
        sg_bidxmaps_pl = tf.placeholder( tf.int32,shape= (batch_size,) + sg_bidxmaps_shape )
        flatten_bidxmaps_pl = tf.placeholder(tf.int32,shape= (batch_size,)+flatten_bidxmaps_shape[0:-1]+(2,),name="flatten_bidxmaps_pl")
        fbmap_neighbor_idis_pl = tf.placeholder(tf.float32,shape= (batch_size,)+flatten_bidxmaps_shape[0:-1]+(1,),name="fbmap_neighbor_idis_pl")

        return pointclouds_pl, labels_pl, smpws_pl,  sg_bidxmaps_pl, flatten_bidxmaps_pl, fbmap_neighbor_idis_pl, sgf_config_pls

def get_sa_module_config(model_flag):
    cascade_num = int(model_flag[0])
    mlps_0 = []
    if model_flag=='1a' or model_flag=='1aG':
        #mlps_0.append( [64,64,128,128,512,1024] )
        mlps_0.append( [64,64,64,128,512] )
    elif model_flag=='1b' or model_flag=='1bG':
        mlps_0.append( [32, 64,64,128,128,256,512] )
    elif model_flag=='2a' or model_flag=='2aG':
        mlps_0.append( [32,64,64,128] )
        mlps_0.append( [128,128,256,512] )
    elif model_flag=='3a' or model_flag=='3aG':
        mlps_0.append( [32,32,64] )
        mlps_0.append( [64,128,256] )
        mlps_0.append( [256,256,512] )
    elif model_flag=='4a' or model_flag=='4aG':
        mlps_0.append( [32,32,64] )
        mlps_0.append( [64,64,128] )
        mlps_0.append( [128,128,256] )
        mlps_0.append( [256,256,512] )
    elif model_flag=='4bG':
        mlps_0.append( [24,24,48] )
        mlps_0.append( [48,48,64] )
        mlps_0.append( [64,64,128] )
        mlps_0.append( [128,128,256] )
    elif model_flag=='5aG':
        mlps_0.append( [32,32,64] )
        mlps_0.append( [64,64,128] )
        mlps_0.append( [128,128,256] )
        mlps_0.append( [256,256,256] )
        mlps_0.append( [256,512,512] )
    elif model_flag=='5bG':
        mlps_0.append( [32,32,48] )
        mlps_0.append( [48,48,64] )
        mlps_0.append( [64,64,128] )
        mlps_0.append( [128,128,256] )
        mlps_0.append( [256,512,512] )

    elif model_flag=='1DSa' or model_flag=='1DSaG':
        dense_config = {}
        dense_config['num_block'] = 1
        dense_config['growth_rate'] = 32
        dense_config['initial_feature_num']=int(dense_config['growth_rate']*1.5)
        dense_config['layers_per_block'] = 5
        dense_config['keep_prob'] = 0.3
        mlps_0.append( dense_config )

    elif model_flag=='4DSa' or model_flag=='4DSaG':
        dense_config = {}
        dense_config['num_block'] = 1
        dense_config['growth_rate'] = 16
        #dense_config['initial_feature_num']=int(dense_config['growth_rate']*1)
        dense_config['layers_per_block'] = 4
        dense_config['keep_prob'] = 0.3
        mlps_0.append( copy.deepcopy(dense_config) )

        #del dense_config['initial_feature_num']
        #dense_config['initial_feature_rate'] = 0.8
        dense_config['layers_per_block'] = 3
        mlps_0.append( dense_config )
        mlps_0.append( dense_config )
        mlps_0.append( dense_config )
    else:
        assert False,"model_flag not recognized: %s"%(model_flag)

    mlps_1 = []
    if model_flag=='1a' or model_flag=='1aG':
        mlps_1.append( [512,256,128] )
    elif model_flag=='1b' or model_flag=='1bG':
        mlps_1.append( [512, 512, 512] )
    elif model_flag=='1DSa' or model_flag=='1DSaG':
        dense_config = {}
        dense_config['num_block'] = 1
        dense_config['growth_rate'] = 32
        dense_config['layers_per_block'] = 2
        dense_config['transition_feature_rate'] = 1
        dense_config['keep_prob'] = 0.3
        mlps_1.append( dense_config )
    else:
        for k in range(cascade_num):
            mlps_1.append( [] )

    return mlps_0, mlps_1

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
    elif model_flag=='4a' or model_flag=='4aG' or model_flag=='4DSaG':
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

def get_model(modelf_nein, rawdata, is_training, num_class, sg_bidxmaps, flatten_bidxmaps, fbmap_neighbor_dis, sgf_configs, sgf_config_pls, bn_decay=None, IsDebug=False):
    """
        rawdata: (B, global_num_point, 6)   (xyz is at first 3 channels)
        out: (N,n1,n2,class)
    """
    IsShowModel = True
    model_flag, num_neighbors = modelf_nein.split('_')
    num_neighbors = np.array( [ int(n) for n in num_neighbors ] )
    assert num_neighbors[0] <= sgf_configs['flatbxmap_max_nearest_num'][0], "There is not enough neighbour indices generated in bxmh5"
    assert num_neighbors[1] <= sgf_configs['flatbxmap_max_nearest_num'][0], "There is not enough neighbour indices generated in bxmh5"
    assert num_neighbors[2] <= np.min(sgf_configs['flatbxmap_max_nearest_num'][1:]), "There is not enough neighbour indices generated in bxmh5"

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
    mlps_0, mlps_1 = get_sa_module_config(model_flag)
    l_points = []                       # size = l_points+1
    l_points.append( rawdata )
    l_xyz = rawdata[...,0:3]     # (2, 512, 128, 6)
    debug = {}
    if IsDebug:
        debug['l_xyz'] = []
        debug['l_xyz'].append( l_xyz )
        debug['grouped_xyz'] = []
        debug['flat_xyz'] = []
        debug['flatten_bidxmap'] = []

    if IsShowModel: print('\n\ncascade_num:%d \ngrouped_rawdata:%s'%(cascade_num, shape_str([rawdata]) ))
    for k in range(cascade_num):
        if IsAddGlobalLayer and k==cascade_num-1:
            IsExtraGlobalLayer = True
            sg_bidxmap_k = None
            block_center_xyz_mm = None
        else:
            IsExtraGlobalLayer = False
            start = sg_bm_extract_idx[k]
            end = sg_bm_extract_idx[k+1]
            sg_bidxmap_k = sg_bidxmaps[ :,start[0]:end[0],0:end[1] ]
            block_center_xyz_mm = sg_bidxmaps[ :,start[0]:end[0],end[1]:end[1]+3 ]
            #block_center_xyz_mm = tf.identity( block_center_xyz_mm, str(k)+"block_center_xyz_mm" )
            # gpu_0/0block_center_xyz_mm:0
            # gpu_0/1block_center_xyz_mm:0
            # gpu_0/2block_center_xyz_mm:0

        if TMPDEBUG:
            pooling = '3DCNN'

        l_xyz, new_points, root_point_features, grouped_xyz = pointnet_sa_module(k, IsExtraGlobalLayer, l_xyz, l_points[k], sg_bidxmap_k,  mlps_0[k], mlps_1[k], block_center_xyz_mm,
                                                                                 sgf_configs,sgf_config_pls, is_training=is_training, bn_decay=bn_decay, pooling=pooling, scope='sa_layer'+str(k) )
        if IsDebug:
            debug['l_xyz'].append( l_xyz )
            debug['grouped_xyz'].append( grouped_xyz )
        if k == 0:
            l_points[0] = root_point_features
        l_points.append(new_points)

        # l_xyz: (2, 512, 128, 6) (2, 512, 3)  (2, 256, 3) (2, 64, 3)
        # l_points: None  (2, 512, 64) (2, 256, 256) (2, 64, 512)
        if IsShowModel: print('pointnet_sa_module %d, l_xyz: %s'%(k,shape_str([l_xyz])))
    if IsShowModel: print('\nafter pointnet_sa_module, l_points:\n%s'%(shape_str(l_points)))
    end_points['l0_points'] = l_points[0]

    # Feature Propagation layers
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
    if IsShowModel:
        print('net:%s'%(shape_str([net])))

    return net, end_points, debug

def get_loss(pred, label, smpw, label_eles_idx ):
    """ pred: BxNxC,
        label: BxN,
	smpw: BxN """
    category_idx = label_eles_idx['label_category'][0]
    label_category = label[...,category_idx]
    smpw_category = smpw[...,category_idx]
    input_drop_mask = tf.get_default_graph().get_tensor_by_name('input_drop_mask/cond/Merge:0')
    if len(input_drop_mask.get_shape()) != 0:
        input_drop_mask = tf.squeeze( input_drop_mask,2 )
        smpw_category = smpw_category * input_drop_mask

    #classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label_category, logits=pred)
    classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label_category, logits=pred, weights=smpw_category)
    tf.summary.scalar('classify loss', classify_loss)
    #tf.add_to_collection('losses',classify_loss)
    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,3))
        net, _ = get_model(inputs, tf.constant(True), 10)
        print(net)
