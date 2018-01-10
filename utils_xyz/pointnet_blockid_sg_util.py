# xyz Dec 2017
# Do 3d point cloud  sample and group by block index
import tensorflow as tf
from block_data_pre_util import GlobalSubBaseBLOCK


def pointnet_sa_module(cascade_id, xyz, points, bidmap, mlp, mlp2, is_training, bn_decay,scope,bn=True,pooling='max', tnet_spec=None, use_xyz=True):
    '''
    grouped_xyz: (batch_size,nsubblock0,npoint_subblock0,3/6/...)
    points: (batch_size,nsubblock0,npoint_subblock0,channel)
    bidmap: (nsubblock1,npoint_subblock1)

    grouped_xyz: (batch_size,nsubblock,npoint_subblock,3)
    group_points: (batch_size,nsubblock,npoint_subblock,channel)

    output:
    new_points: (batch_size,nsubblock,channel)
    '''
    with tf.variable_scope(scope) as sc:

        if cascade_id == 0:
            # already grouped, no need to group
            grouped_xyz = xyz
            grouped_points = xyz

            # the first step, xyz can be actually rawdata include color ...
            # assume xyz in at the first 3 channels
            grouped_xyz_pure = grouped_xyz[...,0:3]
            new_xyz = tf.reduce_mean(grouped_xyz_pure,-1)
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
        else:
            grouped_xyz = tf.gather(xyz,bidmap,axis=1)
            grouped_points = tf.gather(points,bidmap,axis=1)
            # use the average position as new xyz
            new_xyz = tf.reduce_mean(grouped_xyz,-1)

            if use_xyz:
                grouped_points = tf.concat([grouped_xyz,grouped_points],axis=-1)

        nsample = grouped_points.get_shape()[2].value  # the conv kernel size
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(grouped_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv%d'%(i), bn_decay=bn_decay)

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

        if mlp2 is None: mlp2 = []
        for i, num_out_channel in enumerate(mlp2):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv_post_%d'%(i), bn_decay=bn_decay)
        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])




        return new_xyz, new_points


def pointnet_fp_model( points1, points2, flatten_bidxmap, mlp, is_training, bn_decay, scope, bn=True ):
    '''
    in Qi's code, 3 larger balls are weighted back-propogated to one point
    Here, I only back-propogate one
    '''
    mapped_points2 = tf.gather(points2,flatten_bidxmap,axis=1)
    if points1 is not None:
        new_points1 = tf.concat(values=[points1,mapped_points2],axis=-1)
    else:
        new_points1 = mapped_points2
    new_points1 = tf.expand_dims(new_points1,2)

    for i, num_out_channel in enumerate(mlp):
        new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=bn, is_training=is_training,
                                    scope='conv%d'%(i), bn_decay=bn_decay)
    new_points1 = tf.squeeze(new_points1,[2])
    return new_points1
