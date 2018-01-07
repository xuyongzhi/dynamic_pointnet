# xyz Dec 2017
# Do 3d point cloud  sample and group by block index
import tensorflow as tf
from block_data_pre_util import GlobalSubBaseBLOCK


#def pointnet_sa_module(rawdata, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', tnet_spec=None, knn=False, use_rawdata=True):
def pointnet_sa_module(rawdata,points, base_bids_in_new, mlp,mlp2,is_training,bn_decay,scope,bn=True,pooling='max', tnet_spec=None, use_rawdata=True):
    '''
    rawdata: (batch_size,nsubblock0,3/6/...)
    points: (batch_size,nsubblock0,channel)
    base_bids_in_new: (nsubblock1,npoint_subblock1)

    ** 1 cascade, rawdata is grouped_rawdata, base_bids_in_new is None

    grouped_rawdata: (batch_size,nsubblock,npoint_subblock,3)
    group_points: (batch_size,nsubblock,npoint_subblock,channel)

    output:
    new_points: (batch_size,nsubblock,channel)
    '''
    with tf.variable_scope(scope) as sc:
        if base_bids_in_new is None:
            grouped_points = points
            grouped_rawdata = rawdata
        else:
            grouped_rawdata = tf.gather(rawdata,base_bids_in_new,axis=1)
            grouped_points = tf.gather(points,base_bids_in_new,axis=1)

        if grouped_points is None:
            grouped_points = grouped_rawdata
        else:
            if use_rawdata:
                grouped_points = tf.concat([rawdata,grouped_points],axis=-1)

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
                dists = tf.norm(grouped_rawdata,axis=-1,ord=2,keep_dims=True)
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

        return new_points

