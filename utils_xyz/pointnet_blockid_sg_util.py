# xyz Dec 2017
# Do 3d point cloud  sample and group by block index
import tensorflow as tf
import os,sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR+'/../utils')
from block_data_prep_util import GlobalSubBaseBLOCK
import tf_util
import numpy as np

DEBUG_TMP = True
'''
Checking list:
    new_xyz
'''

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

def pointnet_sa_module(cascade_id, IsExtraGlobalLayer, xyz, points, bidmap, mlps_0, mlps_0s_1, block_bottom_center_mm, sgf_configs, sgf_config_pls,
                       is_training, bn_decay,scope,bn=True,pooling='max', tnet_spec=None, use_xyz=True):
    '''
    Input cascade_id==0:
        xyz is grouped_points: (batch_size,nsubblock0,npoint_subblock0,6)
        points: None
        bidmap: None
    Input cascade_id==1:
        xyz: (batch_size,nsubblock0,3)
        points: (batch_size,nsubblock0,channel)
        bidmap: (batch_size,nsubblock1,npoint_subblock1)
    Medium cascade_id==1:
        grouped_xyz: (batch_size,nsubblock1,npoint_subblock1,3)
        new_xyz: (batch_size,nsubblock1,3)
        group_points: (batch_size,nsubblock1,npoint_subblock1,channel)

    output cascade_id==1:
        new_xyz: (batch_size,nsubblock1,3)
        new_points: (batch_size,nsubblock1,channel)
    '''
    IsShowModel = True
    block_bottom_center_mm = tf.cast(block_bottom_center_mm, tf.float32, name='block_bottom_center_mm') # gpu_0/sa_layer3/block_bottom_center_mm:0
    with tf.variable_scope(scope) as sc:
        cascade_num = sgf_configs['flatten_bm_extract_idx'].shape[0]-1
        assert sgf_configs['sub_block_step_candis'].size == cascade_num
        if cascade_id==0:
            input_drop_mask = tf.get_default_graph().get_tensor_by_name('input_drop_mask/cond/Merge:0')

        if IsExtraGlobalLayer:
            #if cascade_id == 0:
            #    points = tf.gather_nd( xyz,flatten_bidxmap0_concat)
            #    xyz = points[...,0:3]
            grouped_xyz = tf.expand_dims( xyz,axis=1 )
            grouped_points = tf.expand_dims(points,axis=1)
            if cascade_id==0 and  len(input_drop_mask.get_shape()) != 0:
                grouped_indrop_mask = tf.expand_dims( input_drop_mask, axis=1, name='grouped_indrop_mask' )
        else:
            assert len(xyz.shape) == 3
            batch_size = xyz.get_shape()[0].value

            batch_idx = tf.reshape( tf.range(batch_size),[batch_size,1,1,1] )
            nsubblock = bidmap.get_shape()[1].value
            npoint_subblock = bidmap.get_shape()[2].value
            batch_idx_ = tf.tile( batch_idx,[1,nsubblock,npoint_subblock,1] )
            bidmap = tf.expand_dims( bidmap,axis=-1 )
            bidmap_concat = tf.concat( [batch_idx_,bidmap],axis=-1 )

            grouped_xyz = tf.gather_nd(xyz,bidmap_concat)
            grouped_points = tf.gather_nd(points,bidmap_concat)
            if cascade_id==0 and  len(input_drop_mask.get_shape()) != 0:
                grouped_indrop_mask = tf.gather_nd( input_drop_mask, bidmap_concat, name='grouped_indrop_mask' )
            # use the average position as new xyz

            if use_xyz:
                grouped_points = tf.concat([grouped_xyz,grouped_points],axis=-1)

        if sgf_configs['mean_grouping_position'] and (not pooling=='3DCNN'):
            new_xyz = tf.reduce_mean(grouped_xyz,-2)
        else:
            new_xyz = block_bottom_center_mm[:,:,3:6] * tf.constant( 0.001, tf.float32 )
        nsample = grouped_points.get_shape()[2].value  # the conv kernel size

        if IsShowModel:
            print('\n\npointnet_sa_module cascade_id:%d\n xyz:%s\n grouped_xyz:%s\n new_xyz:%s\n grouped_points:%s\n nsample:%d'%(
                    cascade_id, shape_str([xyz]), shape_str([grouped_xyz]), shape_str([new_xyz]), shape_str([grouped_points]), nsample))

        new_points = grouped_points

        if 'growth_rate'in mlps_0:
            new_points = tf_util.dense_net( new_points, mlps_0, bn, is_training, bn_decay, scope = 'dense_cascade_%d_point_encoder'%(cascade_id) , is_show_model = IsShowModel )
        else:
            for i, num_out_channel in enumerate(mlps_0):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv%d'%(i), bn_decay=bn_decay)
                if IsShowModel:
                    print('point encoder1 %d, new_points:%s'%(i, shape_str([new_points])))

        if cascade_id == 0:
            root_point_features = new_points
            if len(input_drop_mask.get_shape()) != 0:
                new_points = tf.multiply( new_points, grouped_indrop_mask )
        else:
            root_point_features = None

        if pooling == '3DCNN' and cascade_id == 0:
            pooling = 'max'
        if pooling=='avg':
            new_points = tf_util.avg_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='avgpool1')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg1'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlps_0[-1])
                new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        elif pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)
        elif pooling=='min':
            new_points = tf_util.max_pool2d(-1*new_points, [1,nsample], stride=[1,1], padding='VALID', scope='minpool1')
        elif pooling=='max_and_avg':
            avg_points = tf_util.max_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='maxpool1')
            max_points = tf_util.avg_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='avgpool1')
            new_points = tf.concat([avg_points, max_points], axis=-1)
        elif pooling == '3DCNN':
            block_bottom_center_mm = tf.identity( block_bottom_center_mm,'block_bottom_center_mm' )      # gpu_0/sa_layer3/block_bottom_center_mm:0
            c500 = tf.constant([500],tf.float32)
            c1000 = tf.constant([1000],tf.float32)
            c1 = tf.constant([[[1,1,1]]],tf.float32)
            step_last = sgf_configs['sub_block_step_candis'][cascade_id-1] * c1 # gpu_0/sa_layer3/mul_1:0
            stride_last = sgf_configs['sub_block_stride_candis'][cascade_id-1] * c1 # gpu_0/sa_layer3/mul_2:0
            voxel_bottom_xyz_mm = block_bottom_center_mm[:,:,0:3]
            # NOTE: c1=[1,1,1]*0.5 ONLY when the sh5 step is also the same on three dimensions.
            #                      Otherwise, the stride at each cascade may also be changed.
            min_point_bottom_xyz_mm = voxel_bottom_xyz_mm
            min_point_bottom_xyz_mm = tf.expand_dims( min_point_bottom_xyz_mm, -2, name='min_point_bottom_xyz_mm' ) # gpu_0/sa_layer1/min_point_bottom_xyz_mm:0
            grouped_bottom_xyz_mm = grouped_xyz * c1000 - step_last * c500  # gpu_0/sa_layer1/sub_1:0
            point_indices_f = (grouped_bottom_xyz_mm - min_point_bottom_xyz_mm) / (stride_last*c1000)  # gpu_0/sa_layer3/div:0
            point_indices = tf.rint( point_indices_f,'point_indices' )  # gpu_0/sa_layer3/point_indices:0

            # check indice err
            point_indices_err = tf.abs( point_indices - point_indices_f, name='point_indices_err' )     # gpu_0/sa_layer3/point_indices_err:0
            point_indices_maxerr = tf.reduce_max( point_indices_err, name='point_indices_maxerr_xyz' ) # gpu_0/sa_layer3/point_indices_maxerr_xyz:0
            check_point_indices = tf.assert_less( point_indices_maxerr, 1e0, data=[cascade_id, point_indices_maxerr],
                                                 message='point indices in voxel check on cascade %d '%(cascade_id), name='check_point_indices' )
            tf.add_to_collection( 'check', check_point_indices )

            # check max indice
            check_min_indice = tf.assert_less( tf.constant(-1,tf.float32), tf.reduce_min(point_indices), data=[cascade_id,tf.reduce_min(point_indices)], name='check_min_indice' )
            tf.add_to_collection( 'check', check_min_indice )
            if IsExtraGlobalLayer:
                step_cur = (block_bottom_center_mm[:,:,3:6] - block_bottom_center_mm[:,:,0:3]) * tf.constant(0.002,tf.float32)  # gpu_0/sa_layer3/mul_6:0
                max_indice = tf.ceil( ( step_cur[0,0] - step_last[0,0] ) / stride_last[0,0], name='max_indice_global' ) # gpu_0/sa_layer3/max_indice_global:0
                max_indice += c1[0,0]    # ??? !!! ??? Why?: pading?
            else:
                step_cur = sgf_configs['sub_block_step_candis'][cascade_id] * c1
                max_indice_f = ( step_cur - step_last ) / stride_last  # gpu_0/sa_layer3/div_1:0
                max_indice = tf.rint( max_indice_f[0,0], name='max_indice' ) # gpu_0/sa_layer1/max_indice:0
                max_indice_err = tf.reduce_max(tf.abs(max_indice_f - max_indice))
                check_MAX_indice = tf.assert_less( max_indice_err, tf.constant(1e-0), data=[max_indice_err], name='check_MAX_indice' )
                tf.add_to_collection( 'check', check_MAX_indice )

            for i in range(3):
                real_max = tf.reduce_max(point_indices[:,:,:,i])
                check_max_indice = tf.assert_less_equal( real_max - max_indice[i], tf.constant(1e-7), data=[cascade_id, real_max, max_indice[i]], name='check_max_indice_'+str(i) )
                tf.add_to_collection( 'check', check_max_indice )

            point_indices = tf.cast( point_indices, tf.int32 )

            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True)

        if IsShowModel:
            print('after %s pooling, new_points:%s'%( pooling, shape_str([new_points])))


        if 'growth_rate'in mlps_0s_1:
            new_points = tf_util.dense_net( new_points, mlps_0s_1, bn, is_training, bn_decay, scope = 'dense_cascade_%d_block_encoder'%(cascade_id) , is_show_model = IsShowModel )
        else:
            for i, num_out_channel in enumerate(mlps_0s_1):
                new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_post_%d'%(i), bn_decay=bn_decay)
                if IsShowModel:
                    print('point encoder2 %d, new_points:%s'%(i, shape_str([new_points])))
        # (2, 512, 1, 64)
        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlps_0s_1[-1])

        if IsShowModel:
            print('pointnet_sa_module return\n new_xyz: %s\n new_points:%s\n\n'%(shape_str([new_xyz]),shape_str([new_points])))
            #import pdb;pdb.set_trace()
        # (2, 512, 64)
        return new_xyz, new_points, root_point_features, grouped_xyz


def pointnet_fp_module( cascade_id, num_neighbors, points1, points2, flatten_bidxmap, fbmap_neighbor_idis, mlps_e1, mlps_fp, is_training, bn_decay, scope, bn=True, debug=None ):
    '''
    in Qi's code, 3 larger balls are weighted back-propogated to one point
    Here, I only back-propogate one

    Input:
        points1 (cascade_id=2): (2, 256, 256)
        points2 (cascade_id=3): (2, 64, 512)
        flatten_bidxmap: (B,num_point,self.flatbxmap_max_nearest_num,2)
                    [:,:,:,0]: aim_b_index
                    [:,:,:,1]: point_index_in_aimb  (useless when cascade_id>0)
        fbmap_neighbor_idis: (B,num_point,self.flatbxmap_max_nearest_num,1)
                    [:,:,:,2]: index_distance
        mlps_fp: [256,256]
    Output:
        new_points1: (2, 256, 256)
    '''
    IsShowModel = True
    IsDebug = 'flatten_bidxmap' in debug
    if IsShowModel:
        print('\n\npointnet_fp_module %s\n points1: %s\n points2: %s\n flatten_bidxmap: %s\n'%( scope, shape_str([points1]), shape_str([points2]), shape_str([flatten_bidxmap]) ))
    with tf.variable_scope(scope) as sc:
        assert len(flatten_bidxmap.get_shape()) == 4
        if cascade_id == 0:
            # points1 is grouped point features
            assert len(points1.get_shape()) == 4
        else:
            # points1 is flat point features
            assert len(points1.get_shape()) == 3
            assert flatten_bidxmap.shape[1] == points1.shape[1]
        batch_size = points2.get_shape()[0].value
        batch_idx0 = tf.reshape( tf.range(batch_size),[batch_size,1,1,1] )
        point1_num = flatten_bidxmap.get_shape()[1].value


        if cascade_id == 0:
            # convert grouped points1 to flat point features
            #flatten_bidxmap_aimbidx1 = flatten_bidxmap[:,:,0,0:2]  # (2, 256, 1)
            #flatten_bidxmap_aimbidx_concat1 = tf.concat( [batch_idx, flatten_bidxmap_aimbidx1], axis=-1 )
            #points1 = tf.gather_nd(points1, flatten_bidxmap_aimbidx_concat1 )

            num_neighbor0 = num_neighbors[0]
            disw_theta0 = -1.5 # the abs smaller, more smooth
            assert num_neighbor0 <= flatten_bidxmap.shape[2].value
            batch_idx = tf.tile( batch_idx0,[1, point1_num, num_neighbor0 ,1] ) # (2, 256, 1)
            flatten_bidxmap_concat1 = tf.concat( [batch_idx, flatten_bidxmap[:,:,0:num_neighbor0,0:2]], axis=-1 )  # [...,[batch_idx,aimb_idx,point_idx_in_aimb] ]
            points1_nei = tf.gather_nd( points1, flatten_bidxmap_concat1 )
            if num_neighbor0 > 1:
                dis_weight = tf.nn.softmax( fbmap_neighbor_idis[:,:,0:num_neighbor0,:] * disw_theta0, axis=2 )
                points1_nei = tf.multiply( points1_nei, dis_weight )
            points1 = tf.reduce_sum( points1_nei, axis=2 )

        #flatten_bidxmap_aimbidx = flatten_bidxmap[:,:,0,0:1]  # (2, 256, 1)
        #flatten_bidxmap_aimbidx_concat = tf.concat( [batch_idx, flatten_bidxmap_aimbidx],axis=-1 ) # (2, 256, 2)
        #mapped_points2 = tf.gather_nd(points2, flatten_bidxmap_aimbidx_concat) # (2, 256, 512)


        # use the inverse distance weighted sum of 3 neighboured point features
        if cascade_id == 0:
            num_neighbor = num_neighbors[1]
        else:
            num_neighbor = num_neighbors[2]
        assert num_neighbor <= flatten_bidxmap.shape[2].value
        neighbor_method = 'A'
        #-----------------------------------
        if neighbor_method=='A':
            batch_idx = tf.tile( batch_idx0,[1, point1_num, 1 ,1] ) # (2, 256, 1)
            # from distance to weight
            if num_neighbor>1:
                disw_theta = -0.5 # the abs smaller, more smooth
                dis_weight = tf.nn.softmax( fbmap_neighbor_idis * disw_theta, axis=2 )
            for i in range(num_neighbor):
                flatten_bidxmap_aimbidx_concat_i = tf.concat( [batch_idx, flatten_bidxmap[:,:,i:(i+1),0:1]],axis=-1 ) # (2, 256, 2)
                mapped_points2_nei_i = tf.gather_nd(points2, flatten_bidxmap_aimbidx_concat_i) # (2, 256, 512)
                if num_neighbor>1:
                    mapped_points2_nei_i = tf.multiply( mapped_points2_nei_i, dis_weight[:,:,i:(i+1),:] )
                    if i==0:
                        mapped_points2 = mapped_points2_nei_i
                    else:
                        mapped_points2 += mapped_points2_nei_i
                else:
                    mapped_points2 = mapped_points2_nei_i
            mapped_points2 = tf.squeeze( mapped_points2,2 )
        #-----------------------------------
        if neighbor_method=='B':
            batch_idx = tf.tile( batch_idx0,[1, point1_num, num_neighbor ,1] ) # (2, 256, 1)
            flatten_bidxmap_aimbidx_concat = tf.concat( [batch_idx, flatten_bidxmap[:,:,0:num_neighbor,0:1]],axis=-1 ) # (2, 256, 2)
            mapped_points2_nei = tf.gather_nd(points2, flatten_bidxmap_aimbidx_concat) # (2, 256, 512)
            if num_neighbor>1:
                disw_theta = -0.7 # the abs smaller, more smooth
                dis_weight = tf.nn.softmax( fbmap_neighbor_idis * disw_theta, axis=2 )
                mapped_points2_nei = tf.multiply( mapped_points2_nei, dis_weight )
            mapped_points2 = tf.reduce_sum( mapped_points2_nei,2 )
        #-----------------------------------

       # if IsDebug:
       #     debug['distance_%d'%(cascade_id)] = fbmap_neighbor_idis
       #     debug['dis_weight_%d'%(cascade_id)] = dis_weight

        #if IsDebug:
        #    debug['flatten_bidxmap'].append( flatten_bidxmap )
        #    import pdb; pdb.set_trace()  # XXX BREAKPOINT
        #    flatten_bidxmap_aimbidx_concat_ = tf.concat( [batch_idx, flatten_bidxmap[:,:,0:num_neighbor,0:2]],axis=-1 ) # (2, 256, 2)
        #    flat_xyz = tf.gather_nd( debug['grouped_xyz'][cascade_id],flatten_bidxmap_aimbidx_concat_) # (2, 256, 512)
        #    debug['flat_xyz'].append( flat_xyz )

        new_points1 = points1
        new_points1 = tf.expand_dims(new_points1,1)
        if 'growth_rate'in mlps_e1:
            new_points1 = tf_util.dense_net( new_points0, mlps_e1, bn, is_training, bn_decay, scope = 'dense_cascade_%d_fp'%(cascade_id) , is_show_model = IsShowModel )
        else:
            for i, num_out_channel in enumerate(mlps_e1):
                new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_encoder1_%d'%(i), bn_decay=bn_decay)
                if IsShowModel: print('new_points1:%s'%(shape_str([new_points1])))

        mapped_points2 = tf.expand_dims(mapped_points2,1)
        new_points1 = tf.concat(values=[new_points1,mapped_points2],axis=-1)
        if IsShowModel: print('after concat new_points1:%s'%(shape_str([new_points1])))

        if 'growth_rate'in mlps_fp:
            new_points1 = tf_util.dense_net( new_points1, mlps_fp, bn, is_training, bn_decay, scope = 'dense_cascade_%d_fp'%(cascade_id) , is_show_model = IsShowModel )
        else:
            for i, num_out_channel in enumerate(mlps_fp):
                new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv%d'%(i), bn_decay=bn_decay)
                if IsShowModel: print('new_points1:%s'%(shape_str([new_points1])))
        new_points1 = tf.squeeze(new_points1,[1]) # (2, 256, 256)
        if IsShowModel: print('new_points1:%s'%(shape_str([new_points1])));
        #if IsShowModel:  import pdb; pdb.set_trace()
    return new_points1


