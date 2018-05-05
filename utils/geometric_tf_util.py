# May 2018 xyz
import numpy as np
import tensorflow as tf


def tf_Rx( x ):
    # anticlockwise, x: radian
    Rx = tf.Variable(np.zeros((3,3)), dtype=tf.float32, trainable=False )
    Rx = tf.scatter_nd_update( Rx, [[0,0],  [1,1],      [1,2],      [2,1],      [2,2] ],
                                    [1,     tf.cos(x),  tf.sin(x),  -tf.sin(x), tf.cos(x)   ],
                               name = 'Rx')
    return Rx

def tf_Ry( y ):
    # anticlockwise, y: radian
    Ry = tf.Variable(np.zeros((3,3)), dtype=tf.float32, trainable=False )
    Ry = tf.scatter_nd_update( Ry, [[0,0],       [0,2],      [1,1],     [2,0],      [2,2] ],
                                    [tf.cos(y),  -tf.sin(y),    1,      tf.sin(y),  tf.cos(y)   ],
                               name = 'Ry')
    return Ry

def tf_Rz( z ):
    # anticlockwise, z: radian
    Rz = tf.Variable(np.zeros((3,3)), dtype=tf.float32, trainable=False ) # gpu_0/sa_layer1/Rz:0
    Rz = tf.scatter_nd_update( Rz, [[0,0],       [0,1],      [1,0],      [1,1],      [2,2] ],
                                    [tf.cos(z),  tf.sin(z),  -tf.sin(z), tf.cos(z),  1   ],
                               name = 'Rz')
    return Rz

def tf_R1D( angle, axis ):
    if axis == 'x':
        return tf_Rx(angle)
    elif axis == 'y':
        return tf_Ry(angle)
    elif axis == 'z':
        return tf_Rz(angle)
    else:
        raise NotImplementedError

def tf_EulerRotate( angles, order ='zxy' ):
    R = tf.eye(3, dtype=tf.float32)
    for i in range(3):
        R_i = tf_R1D(angles[i], order[i])
        R = tf.matmul( R_i, R )
    return R

