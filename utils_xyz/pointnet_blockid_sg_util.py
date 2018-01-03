# xyz Decc 2017
# Do 3d point cloud  sample and group by block index
import tensorflow as tf
from block_data_pre_util import GlobalSubBaseBLOCK


#def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', tnet_spec=None, knn=False, use_xyz=True):
def pointnet_sa_module(new_points,GlobalSubBaseBlock,mlp,mlp2,is_training,bn_decay,scope,bn=True,pooling='max', tnet_spec=None, knn=False, use_xyz=True):
    '''
    #new_xyz: (batch_size,nsubblock,3)
    grouped_xyz: (batch_size,nsubblock,npoint_subblock,3)
    new_points: (batch_size,nsubblock,npoint_subblock,channel)

    Return:
    new_points: (batch_size,nsubblock,channel)
    '''
    with tf.variable_scope(scope) as sc:
        nsample = GlobalSubBaseBlock.nsubblock
        for i, num_out_channel in enumerate(mlp):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
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
        return new_points

def sample_group_by_blockid(points,corres_cur_blockids):
    '''
        points: (batch_size,nsubblock0,channel)
        corres_cur_blockids: (nsubblock1,npoint_subblock1)

        Return:
        new_points: (batch_size,nsubblock1,npoint_subblock1,channel)
    '''
    new_points_n = corres_cur_blockids.shape[0]
    new_points = tf.gather(points,corres_cur_blockids,axis=1)
    return new_points


