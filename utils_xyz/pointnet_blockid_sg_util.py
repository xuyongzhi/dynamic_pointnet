# xyz Dec 2017
# Do 3d point cloud  sample and group by block index
import tensorflow as tf
import os,sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR+'/../utils')
from block_data_prep_util import GlobalSubBaseBLOCK
import tf_util

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

def pointnet_sa_module(cascade_id, IsExtraGlobalLayer, xyz, points, bidmap, mlps_0, mlps_0s_1, is_training, bn_decay,scope,bn=True,pooling='max', tnet_spec=None, use_xyz=True):
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
    with tf.variable_scope(scope) as sc:
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
        #elif cascade_id == 0:
        #    # already grouped, no need to group
        #    assert len(xyz.shape) == 4
        #    # (2, 512, 128, 6)
        #    grouped_points = xyz
        #    # the first step, xyz can be actually rawdata include color ...
        #    # assume xyz in at the first 3 channels
        #    grouped_xyz = xyz[...,0:3]
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

        new_xyz = tf.reduce_mean(grouped_xyz,-2)
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
    IsShowModel = False
    IsDebug = 'flatten_bidxmap' in debug
    if IsShowModel:
        print('\n\npointnet_fp_module %s\n points1: %s\n points2: %s\n flatten_bidxmap: %s\n'%( scope, shape_str([points1]), shape_str([points2]), shape_str([flatten_bidxmap]) ))
    num_neighbours = [ int(n) for n in num_neighbors ]
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

            num_neighbour0 = num_neighbours[0]
            disw_theta0 = -1.5 # the abs smaller, more smooth
            assert num_neighbour0 <= flatten_bidxmap.shape[2].value
            batch_idx = tf.tile( batch_idx0,[1, point1_num, num_neighbour0 ,1] ) # (2, 256, 1)
            flatten_bidxmap_concat1 = tf.concat( [batch_idx, flatten_bidxmap[:,:,0:num_neighbour0,0:2]], axis=-1 )  # [...,[batch_idx,aimb_idx,point_idx_in_aimb] ]
            points1_nei = tf.gather_nd( points1, flatten_bidxmap_concat1 )
            if num_neighbour0 > 1:
                dis_weight = tf.nn.softmax( fbmap_neighbor_idis[:,:,0:num_neighbour0,:] * disw_theta0, axis=2 )
                points1_nei = tf.multiply( points1_nei, dis_weight )
            points1 = tf.reduce_sum( points1_nei, axis=2 )

        #flatten_bidxmap_aimbidx = flatten_bidxmap[:,:,0,0:1]  # (2, 256, 1)
        #flatten_bidxmap_aimbidx_concat = tf.concat( [batch_idx, flatten_bidxmap_aimbidx],axis=-1 ) # (2, 256, 2)
        #mapped_points2 = tf.gather_nd(points2, flatten_bidxmap_aimbidx_concat) # (2, 256, 512)


        # use the inverse distance weighted sum of 3 neighboured point features
        if cascade_id == 0:
            num_neighbour = num_neighbours[1]
        else:
            num_neighbour = num_neighbours[2]
        assert num_neighbour <= flatten_bidxmap.shape[2].value
        neighbor_method = 'A'
        #-----------------------------------
        if neighbor_method=='A':
            batch_idx = tf.tile( batch_idx0,[1, point1_num, 1 ,1] ) # (2, 256, 1)
            # from distance to weight
            if num_neighbour>1:
                disw_theta = -0.5 # the abs smaller, more smooth
                dis_weight = tf.nn.softmax( fbmap_neighbor_idis * disw_theta, axis=2 )
            for i in range(num_neighbour):
                flatten_bidxmap_aimbidx_concat_i = tf.concat( [batch_idx, flatten_bidxmap[:,:,i:(i+1),0:1]],axis=-1 ) # (2, 256, 2)
                mapped_points2_nei_i = tf.gather_nd(points2, flatten_bidxmap_aimbidx_concat_i) # (2, 256, 512)
                if num_neighbour>1:
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
            batch_idx = tf.tile( batch_idx0,[1, point1_num, num_neighbour ,1] ) # (2, 256, 1)
            flatten_bidxmap_aimbidx_concat = tf.concat( [batch_idx, flatten_bidxmap[:,:,0:num_neighbour,0:1]],axis=-1 ) # (2, 256, 2)
            mapped_points2_nei = tf.gather_nd(points2, flatten_bidxmap_aimbidx_concat) # (2, 256, 512)
            if num_neighbour>1:
                disw_theta = -0.7 # the abs smaller, more smooth
                dis_weight = tf.nn.softmax( fbmap_neighbor_idis * disw_theta, axis=2 )
                mapped_points2_nei = tf.multiply( mapped_points2_nei, dis_weight )
            mapped_points2 = tf.reduce_sum( mapped_points2_nei,2 )
        #-----------------------------------

        if IsDebug:
            debug['distance_%d'%(cascade_id)] = distance
            debug['dis_weight_%d'%(cascade_id)] = dis_weight

        if IsDebug:
            debug['flatten_bidxmap'].append( flatten_bidxmap )
            flatten_bidxmap_aimbidx_concat_ = tf.concat( [batch_idx, flatten_bidxmap[:,:,0,0:2]],axis=-1 ) # (2, 256, 2)
            flat_xyz = tf.gather_nd( debug['grouped_xyz'][cascade_id],flatten_bidxmap_aimbidx_concat_) # (2, 256, 512)
            debug['flat_xyz'].append( flat_xyz )


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





