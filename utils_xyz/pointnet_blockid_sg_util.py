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

def pointnet_sa_module(cascade_id, cascade_num, xyz, points, bidmap, mlp, mlp2, is_training, bn_decay,scope,bn=True,pooling='max', tnet_spec=None, use_xyz=True, flatten_bidxmap0_concat=None):
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
    IsShowModel = False
    with tf.variable_scope(scope) as sc:

        if cascade_id == 0 and cascade_num>1:
            # already grouped, no need to group
            assert len(xyz.shape) == 4
            # (2, 512, 128, 6)
            grouped_points = xyz
            # the first step, xyz can be actually rawdata include color ...
            # assume xyz in at the first 3 channels
            grouped_xyz = xyz[...,0:3]
        elif cascade_id == cascade_num-1:
            if cascade_id == 0:
                points = tf.gather_nd( xyz,flatten_bidxmap0_concat)
            grouped_xyz = tf.expand_dims( points[...,0:3],axis=1 )
            grouped_points = tf.expand_dims(points,axis=1)
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
            # use the average position as new xyz

            if use_xyz:
                grouped_points = tf.concat([grouped_xyz,grouped_points],axis=-1)

        new_xyz = tf.reduce_mean(grouped_xyz,-2)

        nsample = grouped_points.get_shape()[2].value  # the conv kernel size

        if IsShowModel:
            print('\n\npointnet_sa_module cascade_id:%d\n xyz:%s\n grouped_xyz:%s\n new_xyz:%s\n grouped_points:%s\n nsample:%d'%(
                    cascade_id, shape_str([xyz]), shape_str([grouped_xyz]), shape_str([new_xyz]), shape_str([grouped_points]), nsample))

        new_points = grouped_points
        # [32, 32, 64]
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)
            if IsShowModel:
                print('point encoder1 %d, new_points:%s'%(i, shape_str([new_points])))
            # (2, 512, 128, 32)
            # (2, 512, 128, 32)
            # (2, 512, 128, 64)

        if cascade_id == 0:
            root_points = tf.gather_nd( new_points,flatten_bidxmap0_concat,name="root_points")
        else:
            root_points = None

        if pooling=='avg':
            new_points = tf_util.avg_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='avgpool1')
        elif pooling=='weighted_avg':
            with tf.variable_scope('weighted_avg1'):
                dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
                new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
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

        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp2):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv_post_%d'%(i), bn_decay=bn_decay)
            if IsShowModel:
                print('point encoder2 %d, new_points:%s'%(i, shape_str([new_points])))
        # (2, 512, 1, 64)
        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])

        if IsShowModel:
            print('pointnet_sa_module return\n new_xyz: %s\n new_points:%s\n\n'%(shape_str([new_xyz]),shape_str([new_points])))
            import pdb;pdb.set_trace()
        # (2, 512, 64)
        return new_xyz, new_points, root_points


def pointnet_fp_module( points1, points2, flatten_bidxmap, mlp, is_training, bn_decay, scope, bn=True ):
    '''
    in Qi's code, 3 larger balls are weighted back-propogated to one point
    Here, I only back-propogate one

    Input:
        points1 (cascade_id=2): (2, 256, 256)
        points2 (cascade_id=3): (2, 64, 512)
        flatten_bidxmap: (2, 256, 2)
        mlp: [256,256]
    Output:
        new_points1: (2, 256, 256)
    '''
    IsShowModel = False
    if IsShowModel:
        print('\n\npointnet_fp_module %s\n points1: %s\n points2: %s\n flatten_bidxmap: %s\n'%( scope, shape_str([points1]), shape_str([points2]), shape_str([flatten_bidxmap]) ))
    with tf.variable_scope(scope) as sc:
        flatten_bidxmap = flatten_bidxmap[:,:,0:1]  # (2, 256, 1)
        batch_size = points2.get_shape()[0].value
        batch_idx = tf.reshape( tf.range(batch_size),[batch_size,1,1] )
        flatten_bidxmap_shape1 = flatten_bidxmap.get_shape()[1].value
        batch_idx = tf.tile( batch_idx,[1,flatten_bidxmap_shape1,1] ) # (2, 256, 1)

        flatten_bidxmap_concat = tf.concat( [batch_idx,flatten_bidxmap],axis=-1 ) # (2, 256, 2)
        mapped_points2 = tf.gather_nd(points2,flatten_bidxmap_concat) # (2, 256, 512)
        new_points1 = tf.concat(values=[points1,mapped_points2],axis=-1)
        new_points1 = tf.expand_dims(new_points1,2)     # (2, 256, 1, 768)
        if IsShowModel: print('new_points1:%s'%(shape_str([new_points1])))

        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)
            if IsShowModel: print('new_points1:%s'%(shape_str([new_points1])))
        new_points1 = tf.squeeze(new_points1,[2]) # (2, 256, 256)
        if IsShowModel: print('new_points1:%s'%(shape_str([new_points1])));
        if IsShowModel:  import pdb; pdb.set_trace()
    return new_points1
