""" Wrapper functions for TensorFlow layers.

Author: Charles R. Qi
Date: November 2017
"""

import numpy as np
import tensorflow as tf

CNN_CONFIGS = {}
CNN_CONFIGS['norm'] = 'batch'
def show_CNN_CONFIGS():
    print('\nCNN_CONFIGS:')
    print('\tnorm:',CNN_CONFIGS['norm'])

def shape_str(tensor_ls):
    if type(tensor_ls) != list:
        tensor_ls = [tensor_ls]
    shape_str = ''
    for i in range(len(tensor_ls)):
        if tensor_ls[i] == None:
            shape_str += '\t None'
        else:
            shape_str += '\t' + str( [s.value for s in tensor_ls[i].shape] )
        if i < len(tensor_ls)-1:
            shape_str += '\n'
    return shape_str

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        #print( tf.get_variable_scope().name )
        #print( tf.get_variable_scope().reuse )
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 1D convolution with non-linear operation.

  Args:
    inputs: 3-D tensor variable BxLxC
    num_output_channels: int
    kernel_size: int
    scope: string
    stride: int
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    num_in_channels = inputs.get_shape()[-1].value
    kernel_shape = [kernel_size,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    outputs = tf.nn.conv1d(inputs, kernel,
                           stride=stride,
                           padding=padding)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)

    if bn:
        outputs = norm_template( outputs, is_training,
                            bn_decay, scope='bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs




def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(inputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = norm_template( outputs, is_training,
                                bn_decay, scope='bn')

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs



def conv2d_(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
    #! The difference with conv2d is: do norm and activation first!:
    # This is desiged for densenet.
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
      outputs = inputs
      if bn:
        outputs = norm_template( outputs, is_training,
                                bn_decay, scope='bn')
      if activation_fn is not None:
        outputs = activation_fn(outputs)

      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(outputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)
      return outputs


def conv2d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=0.0,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None):
  """ 2D convolution transpose with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor

  Note: conv2d(conv2d_transpose(a, num_out, ksize, stride), a.shape[-1], ksize, stride) == a
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_output_channels, num_in_channels] # reversed to conv2d
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride

      # from slim.convolution2d_transpose
      def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
          dim_size *= stride_size

          if padding == 'VALID' and dim_size is not None:
            dim_size += max(kernel_size - stride_size, 0)
          return dim_size

      # caculate output shape
      batch_size = inputs.get_shape()[0].value
      height = inputs.get_shape()[1].value
      width = inputs.get_shape()[2].value
      out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
      out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
      output_shape = [batch_size, out_height, out_width, num_output_channels]

      outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = norm_template( outputs, is_training,
                                bn_decay, scope='bn')

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs



def conv3d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.0,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None,
           name=None):
  """ 3D convolution with non-linear operation.

  Args:
    inputs: 5-D tensor variable BxDxHxWxC
    num_output_channels: int
    kernel_size: a list of 3 ints
    scope: string
    stride: a list of 3 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    num_in_channels = inputs.get_shape()[-1].value
    kernel_shape = [kernel_d, kernel_h, kernel_w,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.conv3d(inputs, kernel,
                           [1, stride_d, stride_h, stride_w, 1],
                           padding=padding )
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)

    if bn:
        outputs = norm_template( outputs, is_training,
                                bn_decay, scope='bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs, name=name)
    else:
        outputs = tf.identity( outputs, name )
    return outputs

def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.0,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
  """ Fully connected layer with non-linear operation.

  Args:
    inputs: 2-D tensor BxN
    num_outputs: int

  Returns:
    Variable tensor of size B x num_outputs.
  """
  with tf.variable_scope(scope) as sc:
    num_input_units = inputs.get_shape()[-1].value
    weights = _variable_with_weight_decay('weights',
                                          shape=[num_input_units, num_outputs],
                                          use_xavier=use_xavier,
                                          stddev=stddev,
                                          wd=weight_decay)
    outputs = tf.matmul(inputs, weights)
    biases = _variable_on_cpu('biases', [num_outputs],
                             tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)

    if bn:
        outputs = norm_template( outputs, is_training,
                                bn_decay, scope='bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D max pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs

def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D avg pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.avg_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs


def max_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D max pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.max_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs

def avg_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D avg pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.avg_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs





def batch_norm_template_unused(inputs, is_training, scope, moments_dims, bn_decay):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
  Return:
      normed:        batch-normalized maps
  """
  # For support of GAN
  #bn_decay = bn_decay if bn_decay is not None else 0.9
  #return tf.contrib.layers.batch_norm(inputs,
  #                                    center=True, scale=True,
  #                                    is_training=is_training, decay=bn_decay,updates_collections=None,
  #                                    scope=scope)
  with tf.variable_scope(scope) as sc:
    num_channels = inputs.get_shape()[-1].value
    beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                       name='beta', trainable=True)
    gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                        name='gamma', trainable=True)
    batch_mean, batch_var = tf.nn.moments(inputs, moments_dims, name='moments')
    decay = bn_decay if bn_decay is not None else 0.9
    ema = tf.train.ExponentialMovingAverage(decay=decay)
    # Operator that maintains moving averages of variables.
    # Need to set reuse=False, otherwise if reuse, will see moments_1/mean/ExponentialMovingAverage/ does not exist
    # https://github.com/shekkizh/WassersteinGAN.tensorflow/issues/3
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        ema_apply_op = tf.cond(is_training,
                               lambda: ema.apply([batch_mean, batch_var]),
                               lambda: tf.no_op())

    # Update moving average and return current batch's avg and var.
    def mean_var_with_update():
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)

    # ema.average returns the Variable holding the average of var.
    mean, var = tf.cond(is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))
    normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
  return normed

def norm_template( inputs, is_training, bn_decay, scope, data_format='NHWC', norm_type=CNN_CONFIGS['norm'], G=32, esp=1e-5):
    # https://github.com/shaohua0116/Group-Normalization-Tensorflow
    with tf.variable_scope('{}_norm'.format(norm_type)):
        if norm_type == 'none':
            outputs = inputs
        elif norm_type == 'batch':
            outputs = batch_norm_template(inputs, is_training, bn_decay, scope, data_format )
        elif norm_type == 'group':
            outputs = group_norm_template(inputs, is_training, bn_decay, scope, data_format, G, esp)
        else:
            raise NotImplementedError
    return outputs

def group_norm_template(  inputs, is_training, bn_decay, scope, data_format='NHWC', G=32, esp=1e-5 ):
    with tf.variable_scope(scope):
        # normalize
        #if data_format=='NCHW':
        #    inputs = tf.transpose(inputs, [0, 2, 3, 1])
        assert data_format[-1]=='C' # 'NHWC' 'NDHWC'  'NWC'
        org_shape = inputs.get_shape().as_list()
        dim_n = len(org_shape)
        C = org_shape[-1]
        G = min(G, C)
        assert C % G ==0, "Group norm: C=%d, G=%d, not integrally divided"%(C, G)
        grouped_shape = org_shape[0:dim_n-1] + [G, C//G]  # [N, H, W, G, C // G]

        inputs = tf.reshape(inputs, grouped_shape)
        moments_dims = range(1, dim_n-1) + [dim_n]   # except batch_dim and group_dim
            # [1, 2, 4] for NHWC
        mean, var = tf.nn.moments(inputs, moments_dims, keep_dims=True)
        outputs = (inputs - mean) / tf.sqrt(var + esp)
        # per channel gamma and beta
        gamma = tf.get_variable('gamma', [C],
                                initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [C],
                                initializer=tf.constant_initializer(0.0))
        gamma = tf.reshape(gamma, [1]*(dim_n-1) + [C])
        beta = tf.reshape(beta, [1]*(dim_n-1) + [C])

        outputs = tf.reshape( outputs, org_shape ) * gamma + beta

        #if data_format=='NCHW':
        #    outputs = tf.transpose(inputs, [0, 3, 1, 2])
        return outputs


def batch_norm_template(inputs, is_training, bn_decay,  scope, data_format='NHWC'):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
      data_format:   'NHWC' or 'NCHW'
  Return:
      normed:        batch-normalized maps
  """
  bn_decay = bn_decay if bn_decay is not None else 0.9
  return tf.contrib.layers.batch_norm(inputs,
                                      center=True, scale=True,
                                      is_training=is_training, decay=bn_decay,updates_collections=None,
                                      scope=scope,
                                      data_format=data_format)

def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None,
            name=None):
  """ Dropout layer.

  Args:
    inputs: tensor
    is_training: boolean tf.Variable
    scope: string
    keep_prob: float in [0,1]
    noise_shape: list of ints

  Returns:
    tensor variable
  """
  with tf.variable_scope(scope) as sc:
    outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs,
                      name=name)
    return outputs


def dense_net( inputs, dense_config, bn, is_training, bn_decay, activation_fn=tf.nn.relu, scope='dense_net', is_show_model = False ):
    assert 'growth_rate' in dense_config
    assert 'layers_per_block' in dense_config
    assert 'keep_prob' in dense_config # 0.9
    assert 'num_block' in dense_config
    if dense_config['num_block'] > 1:
        assert 'transition_feature_rate' in dense_config # 1

    with tf.variable_scope(scope):
        if is_show_model:
            print('%s \ninputs:\t%s'%(scope, shape_str(inputs)))
        if 'initial_feature_num' in dense_config or 'initial_feature_rate' in dense_config:
            if 'initial_feature_num' in dense_config:
                initial_feature_num = dense_config['initial_feature_num']
                assert 'initial_feature_rate' not in dense_config
            elif 'initial_feature_rate' in dense_config:
                initial_feature_num = int(dense_config['initial_feature_rate'] * inputs.get_shape()[-1].value)

            outputs = conv2d_(inputs, initial_feature_num, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=False, is_training=is_training,
                                        scope='conv_initial', bn_decay=bn_decay, activation_fn=None)
            if is_show_model:
                print('initial conv: \t%s'%( shape_str(outputs)))
        else:
            outputs = inputs
        for b in range( dense_config['num_block'] ):
            # add layer to a block
            for i in range( dense_config['layers_per_block'] ):
                output_bi = conv2d_(outputs, dense_config['growth_rate'], [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_b%d_l%d'%(b,i), bn_decay=bn_decay, activation_fn=activation_fn)
                output_bi = dropout( output_bi, is_training, scope='drop_b%d_l%d'%(b,i), keep_prob=dense_config['keep_prob'] )
                outputs = tf.concat( [outputs,output_bi],-1 )
                if is_show_model:
                    print('b%d l%d conv: \t%s \t%s'%(b,i, shape_str(output_bi), shape_str(outputs)))
            # trainsition layer: not exist in last block
            if b != dense_config['num_block']-1:
                feature_num = int( outputs.get_shape()[-1].value * dense_config['transition_feature_rate'])
                outputs = conv2d_(outputs, feature_num, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_transition%d'%(b), bn_decay=bn_decay, activation_fn=activation_fn)
                outputs = dropout( outputs, is_training, scope='drop_transi_b%d'%(b), keep_prob=dense_config['keep_prob'] )
                if is_show_model:
                    print('b%d transi conv: \t%s'%(b,shape_str(outputs)))
        if bn:
            outputs = norm_template( outputs, is_training,
                                    bn_decay, scope='bn')
        if activation_fn is not None:
            outputs = activation_fn(outputs)

    return outputs
