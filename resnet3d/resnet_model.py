
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for Residual Networks.

Residual networks ('v1' ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR,'utils'))

DEBUG_TMP = False


_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


def get_voxel3dcnn_sa_config( model_flag ):
    cascade_num = int(model_flag[0])
    mlp_pe = []
    mlp_be = []
    # The first cascade is pointnet, so voxel parameters are []
    voxel_channels = [[]]
    voxel_kernels = [[]]
    voxel_strides = [[]]
    voxel_paddings = [[]]

    if model_flag=='3Vm':
        mlp_pe.append( [64,64,128] )

        voxel_channels.append( [128,128,256] )
        voxel_kernels.append( [3,3,3] )
        voxel_strides.append( [1,1,1] )
        voxel_paddings.append( [1,1,0] )

        voxel_channels.append( [256,256,512,1024] )
        voxel_kernels.append( [3,3,3,3] )
        voxel_strides.append( [1,1,1,1] )
        voxel_paddings.append( [1,1,1,0] )


    for l in range(cascade_num-1):
        mlp_pe.append([])
    for l in range(cascade_num): # not used currently
        mlp_be.append([])

    mlp_configs = {}
    mlp_configs['voxel_channels'] = voxel_channels
    mlp_configs['voxel_kernels'] = voxel_kernels
    mlp_configs['voxel_strides'] = voxel_strides
    mlp_configs['voxel_paddings'] = voxel_paddings

    mlp_configs['point_encoder'] = mlp_pe
    mlp_configs['block_learning'] = '3DCNN'
    mlp_configs['block_encoder'] = mlp_be

    assert len(mlp_pe[0]) >0
    assert len(voxel_channels[0])==0
    return mlp_configs
def get_sa_module_config(model_flag):
    if '-S' in model_flag:
        tmp = model_flag.split('-S')
        assert len(tmp) == 2
        model_flag = tmp[0]

        tmp = tmp[1]
        if 'L' in tmp:
            assert False,  "loss_scale_num aborted"
            assert len(tmp) == 3
            scale_num = int(tmp[0])
            loss_scale_num = int(tmp[2])
        else:
            assert len(tmp) == 1
            scale_num = int(tmp)
            loss_scale_num = 1
    else:
        scale_num = 1
        loss_scale_num = 1


    if model_flag[1] == 'V':
        mlp_configs = get_voxel3dcnn_sa_config(model_flag)
    else:
        mlp_configs = get_pointmax_sa_config(model_flag)

    mlp_configs['scale_channel'] = 256      # for multi-scale classification task only
    mlp_configs['scale_num'] = scale_num
    mlp_configs['loss_scale_num'] = loss_scale_num
    return mlp_configs

################################################################################
# xyz add
def tensor_info(tensor_ls, tensor_name_ls=None, scope=None):
  if type(tensor_ls) != list:
    tensor_ls = [tensor_ls]
  if tensor_name_ls == None:
    tensor_name_ls = [''] * len(tensor_ls)
  elif type(tensor_name_ls) != list:
    tensor_name_ls = [tensor_name_ls]
  tensor_info = ''
  for i in range(len(tensor_ls)):
    if scope!=None:
      tensor_info += '\t {} '.format(scope)
    tensor_info += '%-20s:\t'%(tensor_name_ls[i])
    if tensor_ls[i] == None:
        tensor_info += 'None'
    else:
        tensor_info += str( [s.value for s in tensor_ls[i].shape] )
    if i < len(tensor_ls)-1:
        tensor_info += '\n'
  return tensor_info

def unique_nd( inputs, axis=-1, unit=3 ):
    org_inputs = inputs
    org_shape = inputs.shape
    batch_size = org_shape[0].value
    block_num = org_shape[1].value
    point_num = org_shape[2].value
    assert org_shape[3].value == 3

    units = tf.constant( [[9],[3],[1]], tf.float32 )
    inputs = tf.identity( inputs, name='uni_in0' ) # gpu_0/sa_layer4/uni_in0:0
    inputs = tf.reshape( inputs, [batch_size*block_num, point_num,3] )
    first_unique_masks = []
    for i in range(batch_size*block_num):
        inputs_i = tf.reshape( inputs[i], [-1,3], name='uni_inb_%d'%(i) ) # gpu_0/sa_layer4/uni_inb_0:0
        ids = tf.squeeze( tf.matmul( inputs_i, units, name='ids_%d'%(i) ))
        ids_unique, idx_unique = tf.unique( ids, name='idx_unique_%d'%(i) ) # gpu_0/sa_layer4/idx_unique_0:0  gpu_0/sa_layer4/idx_unique_0:1
        is_the_first = idx_unique[1:] - idx_unique[0:idx_unique.shape[0]-1]
        is_the_first = tf.concat( [tf.constant([1],tf.int32),is_the_first],0, name='is_the_first_%d'%(i) ) # gpu_0/sa_layer4/is_the_first_0:0
        first_unique_mask = tf.equal( is_the_first, 1, name='first_unique_mask_%d'%(i) ) # gpu_0/sa_layer4/first_unique_mask_0:0
        first_unique_masks.append( tf.expand_dims(first_unique_mask,0) )
    first_unique_masks = tf.concat( first_unique_masks, 0)
    first_unique_masks = tf.reshape( first_unique_masks, org_shape[0:3], name='first_unique_masks' )
    # set all the replicated items as -9999
    first_unique_masks = tf.expand_dims( first_unique_masks,-1 )
    first_unique_masks = tf.tile( first_unique_masks, [1,1,1,3] )
    output = tf.where( first_unique_masks, org_inputs, tf.ones(org_shape,tf.float32)*(-99), name='uni_out' ) # gpu_0/sa_layer4/uni_out:0
    return output, first_unique_masks

################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs, training, data_format):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, fused=True)


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)

def conv3d_fixed_padding(inputs, filters, kernel_size, strides, padding, data_format):
  """Strided 3-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  if strides > 1:
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    inputs = fixed_padding(inputs, kernel_size, data_format)

  return tf.layers.conv3d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=padding, use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(),
      data_format=data_format)

################################################################################
# ResNet block definitions.
################################################################################
def _building_block_v1(inputs, filters, training, projection_shortcut, strides,
                       data_format):
  """A single block for ResNet v1, without a bottleneck.

  Convolution then batch normalization then ReLU as described by:
    Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = batch_norm(inputs=shortcut, training=training,
                          data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs


def _building_block_v2(inputs, filters, training, projection_shortcut, strides,
                       data_format):
  """A single block for ResNet v2, without a bottleneck.

  Batch normalization then ReLu then convolution as described by:
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/pdf/1603.05027.pdf
    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  display = True

  shortcut = inputs
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  if display: print( tensor_info(inputs, 'conv2d', 'block_v2') )

  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)
  if display: print( tensor_info(inputs, 'conv2d', 'block_v2')+'\n' )

  return inputs + shortcut


def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,
                         strides, data_format):
  """A single block for ResNet v1, with a bottleneck.

  Similar to _building_block_v1(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    shortcut = batch_norm(inputs=shortcut, training=training,
                          data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs


def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,
                         strides, data_format):
  """A single block for ResNet v2, without a bottleneck.

  Similar to _building_block_v2(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Adapted to the ordering conventions of:
    Batch normalization then ReLu then convolution as described by:
      Identity Mappings in Deep Residual Networks
      https://arxiv.org/pdf/1603.05027.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  shortcut = inputs
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  # The projection shortcut should come after the first batch norm and ReLU
  # since it performs a 1x1 convolution.
  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)

  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)

  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)
  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format)

  return inputs + shortcut


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """

  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters * 4 if bottleneck else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format)

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                    data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, training, None, 1, data_format)

  return tf.identity(inputs, name)


class Model(object):
  """Base class for building the Resnet Model."""

  def __init__(self, model_flag, resnet_size, bottleneck, num_classes, num_filters,
               kernel_size,
               conv_stride, first_pool_size, first_pool_stride,
               block_sizes, block_strides,
               final_size, resnet_version=DEFAULT_VERSION, data_format=None,
               dtype=DEFAULT_DTYPE, data_paras={}):
    """Creates a model for classifying an image.

    Args:
      resnet_size: A single integer for the size of the ResNet model.
      bottleneck: Use regular blocks or bottleneck blocks.
      num_classes: The number of classes used as labels.
      num_filters: The number of filters to use for the first block layer
        of the model. This number is then doubled for each subsequent block
        layer.
      kernel_size: The kernel size to use for convolution.
      conv_stride: stride size for the initial convolutional layer
      first_pool_size: Pool size to be used for the first pooling layer.
        If none, the first pooling layer is skipped.
      first_pool_stride: stride size for the first pooling layer. Not used
        if first_pool_size is None.
      block_sizes: A list containing n values, where n is the number of sets of
        block layers desired. Each value should be the number of blocks in the
        i-th set.
      block_strides: List of integers representing the desired stride size for
        each of the sets of block layers. Should be same length as block_sizes.
      final_size: The expected size of the model after the second pooling.
      resnet_version: Integer representing which version of the ResNet network
        to use. See README for details. Valid values: [1, 2]
      data_format: Input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
      dtype: The TensorFlow dtype to use for calculations. If not specified
        tf.float32 is used.

    Raises:
      ValueError: if invalid version is selected.
    """
    self.model_flag = model_flag
    self.resnet_size = resnet_size

    if not data_format:
      data_format = (
          'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
    data_format = 'channels_last'

    self.resnet_version = resnet_version
    if resnet_version not in (1, 2):
      raise ValueError(
          'Resnet version should be 1 or 2. See README for citations.')

    self.bottleneck = bottleneck
    if bottleneck:
      if resnet_version == 1:
        self.block_fn = _bottleneck_block_v1
      else:
        self.block_fn = _bottleneck_block_v2
    else:
      if resnet_version == 1:
        self.block_fn = _building_block_v1
      else:
        self.block_fn = _building_block_v2

    if dtype not in ALLOWED_TYPES:
      raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

    self.data_format = data_format
    self.num_classes = num_classes
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.conv_stride = conv_stride
    self.first_pool_size = first_pool_size
    self.first_pool_stride = first_pool_stride
    self.block_sizes = block_sizes
    self.block_strides = block_strides
    self.final_size = final_size
    self.dtype = dtype
    self.pre_activation = resnet_version == 2
    self.data_paras = data_paras

    self._preprocess_configs()

  def _preprocess_configs(self):
    if '_' in self.model_flag:
        self.model_flag, num_neighbors = modelf_nein.split('_')
        self.num_neighbors = np.array( [ int(n) for n in num_neighbors ] )
    else:
        self.num_neighbors= None
    self.global_numpoint = self.data_paras['points'][0]
    self.cascade_num = int(self.model_flag[0])
    assert self.cascade_num <= self.data_paras['sg_bm_extract_idx'].shape[0]-1
    #print('cascade_num:{}'.format(self.cascade_num))
    self.IsOnlineGlobal = self.model_flag[-1] == 'G'
    self.mlp_configs = get_sa_module_config(self.model_flag)
    for key in self.data_paras:
      setattr(self, key, self.data_paras[key])

    self.IsShowModel = True
    self.mean_grouping_position = True
    self.xyz_elements = ['raw', 'sub_mid', 'global_mid']

  def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                           *args, **kwargs):
    """Creates variables in fp32, then casts to fp16 if necessary.

    This function is a custom getter. A custom getter is a function with the
    same signature as tf.get_variable, except it has an additional getter
    parameter. Custom getters can be passed as the `custom_getter` parameter of
    tf.variable_scope. Then, tf.get_variable will call the custom getter,
    instead of directly getting a variable itself. This can be used to change
    the types of variables that are retrieved with tf.get_variable.
    The `getter` parameter is the underlying variable getter, that would have
    been called if no custom getter was used. Custom getters typically get a
    variable with `getter`, then modify it in some way.

    This custom getter will create an fp32 variable. If a low precision
    (e.g. float16) variable was requested it will then cast the variable to the
    requested dtype. The reason we do not directly create variables in low
    precision dtypes is that applying small gradients to such variables may
    cause the variable not to change.

    Args:
      getter: The underlying variable getter, that has the same signature as
        tf.get_variable and returns a variable.
      name: The name of the variable to get.
      shape: The shape of the variable to get.
      dtype: The dtype of the variable to get. Note that if this is a low
        precision dtype, the variable will be created as a tf.float32 variable,
        then cast to the appropriate dtype
      *args: Additional arguments to pass unmodified to getter.
      **kwargs: Additional keyword arguments to pass unmodified to getter.

    Returns:
      A variable which is cast to fp16 if necessary.
    """

    if dtype in CASTABLE_TYPES:
      var = getter(name, shape, tf.float32, *args, **kwargs)
      return tf.cast(var, dtype=dtype, name=name + '_cast')
    else:
      return getter(name, shape, dtype, *args, **kwargs)

  def _model_variable_scope(self):
    """Returns a variable scope that the model should be created under.

    If self.dtype is a castable type, model variable will be created in fp32
    then cast to self.dtype before being used.

    Returns:
      A variable scope for the model.
    """

    return tf.variable_scope('resnet_model',
                             custom_getter=self._custom_dtype_getter)

  def __call__(self, inputs_dic, training):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """

    return self._call(
          inputs_dic['points'],
          inputs_dic['sg_all_bidxmaps'],
          inputs_dic['bidxmaps_flat'],
          inputs_dic['fmap_neighbor_idis'],
          training)

  def _call(self, inputs, sg_bidxmaps, bidxmaps_flat, fmap_neighbor_idis, is_training):
    self.is_training = is_training
    sg_bm_extract_idx = self.data_paras['sg_bm_extract_idx']

    with self._model_variable_scope():
      l_points = []                       # size = l_points+1
      l_points.append( inputs )
      l_xyz = inputs[...,0:3]
      new_points = inputs

      start = sg_bm_extract_idx[-2]
      end = sg_bm_extract_idx[-1]
      globalb_bottom_center_mm = sg_bidxmaps[ :,start[0]:end[0],end[1]:end[1]+6 ]
      globalb_bottom_center = tf.multiply( tf.cast( globalb_bottom_center_mm, tf.float32), 0.001, name='globalb_bottom_center' ) # gpu_0/globalb_bottom_center
      self.max_step_stride = tf.multiply( globalb_bottom_center[:,:,3:6] - globalb_bottom_center[:,:,0:3], 2.0, name='max_step_stride') # gpu_0/max_step_stride

      full_cascades = sg_bm_extract_idx.shape[0]-1
      scales_feature = []

      for k in range(self.cascade_num):
          IsExtraGlobalLayer = False

          if k==self.cascade_num-1 and self.IsOnlineGlobal:
              sg_bidxmap_k = None
              block_bottom_center_mm = globalb_bottom_center_mm
          else:
              start = sg_bm_extract_idx[k]
              end = sg_bm_extract_idx[k+1]
              sg_bidxmap_k = sg_bidxmaps[ :,start[0]:end[0],0:end[1] ]
              block_bottom_center_mm = sg_bidxmaps[ :,start[0]:end[0],end[1]:end[1]+6 ]

          l_xyz, new_points, root_point_features = self.pointnet_sa_module(k, l_xyz,
                          new_points, sg_bidxmap_k, block_bottom_center_mm, scope='sa_layer'+str(k) )
          if k == 0:
              l_points[0] = root_point_features
          l_points.append(new_points)

      if self.IsShowModel:
          print('\nafter pointnet_sa_module, l_points:\n%s'%(tensor_info(l_points)))

      # ----------------------
      inputs = new_points
      axes = [2] if self.data_format == 'channels_first' else [1]
      inputs = tf.reduce_mean(inputs, axes)
      inputs = tf.identity(inputs, 'final_reduce_mean')
      if self.IsShowModel: print( tensor_info(inputs, 'final_reduce_mean') )

      inputs = tf.layers.dense(inputs=inputs, units=self.num_classes)
      inputs = tf.identity(inputs, 'final_dense')
      if self.IsShowModel: print( tensor_info(inputs, 'final_dense') +'\n\n' )
      return inputs

  def pointnet_sa_module(self, cascade_id, xyz, points, bidmap, block_bottom_center_mm,
                       scope, bn=True, tnet_spec=None, use_xyz=True):
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
    block_bottom_center_mm = tf.cast(block_bottom_center_mm, tf.float32, name='block_bottom_center_mm') # gpu_0/sa_layer3/block_bottom_center_mm:0
    new_xyz, grouped_xyz, new_points, valid_mask = self.grouping(cascade_id, xyz,
                        points, bidmap, block_bottom_center_mm, scope, use_xyz )
    new_points, root_point_features = self.sa_model(cascade_id, new_points, grouped_xyz, valid_mask, block_bottom_center_mm, scope, bn, tnet_spec)
    return new_xyz, new_points, root_point_features

  def grouping(self, cascade_id, xyz, points, bidmap, block_bottom_center_mm,
                       scope, use_xyz=True):
    batch_size = xyz.get_shape()[0].value
    with tf.variable_scope(scope) as sc:
        assert self.cascade_num == self.flatten_bm_extract_idx.shape[0]-1  # include global here (Note: cascade_num does not include global in block_pre_util )
        assert self.sub_block_step_candis.size == self.cascade_num-1
        #if cascade_id==0:
        #    indrop_keep_mask = tf.get_default_graph().get_tensor_by_name('indrop_keep_mask:0') # indrop_keep_mask:0

        assert len(xyz.shape) == 3

        if bidmap==None:
            grouped_xyz = tf.expand_dims( xyz, 1 )
            grouped_points = tf.expand_dims( points, 1 )
            new_xyz = None
            valid_mask = None
        else:
            batch_idx = tf.reshape( tf.range(batch_size),[batch_size,1,1,1] )
            nsubblock = bidmap.get_shape()[1].value
            npoint_subblock = bidmap.get_shape()[2].value
            batch_idx_ = tf.tile( batch_idx,[1,nsubblock,npoint_subblock,1] )
            bidmap = tf.expand_dims( bidmap,axis=-1, name='bidmap' )
            bidmap_concat = tf.concat( [batch_idx_,bidmap],axis=-1, name='bidmap_concat' )  # gpu_0/sa_layer0/bidmap_concat:0
            # The value for invalid item in bidmap is -17.
            # On GPU, the responding grouped_xyz and grouped_points is 0.
            # NOT WORK on CPU !!!

            # invalid indices comes from merge_blocks_while_fix_bmap
            # set point_indices_f for invalid points as
            # NETCONFIG['redundant_points_in_block'] ( shoud be set < -500)
            valid_mask = tf.greater( bidmap, tf.constant(-500,tf.int32), 'valid_mask' ) # gpu_0/sa_layer0/valid_mask:0

            grouped_xyz = tf.gather_nd(xyz, bidmap_concat, name='grouped_xyz')  # gpu_0/sa_layer0/grouped_xyz:0
            grouped_points = tf.gather_nd(points,bidmap_concat, name='group_points')
            #if cascade_id==0 and  len(indrop_keep_mask.get_shape()) != 0:
            #    grouped_indrop_keep_mask = tf.gather_nd( indrop_keep_mask, bidmap_concat, name='grouped_indrop_keep_mask' )  # gpu_0/sa_layer0/grouped_indrop_keep_mask:0

        # new_xyz is the "voxel center" or "mean position of points in the voxel"
        if self.mean_grouping_position and (not self.mlp_configs['block_learning']=='3DCNN'):
            new_xyz = tf.reduce_mean(grouped_xyz,-2)
        else:
            new_xyz = block_bottom_center_mm[:,:,3:6] * tf.constant( 0.001, tf.float32 )
        # the mid can be mean or block center, decided by configs['mean_grouping_position']
        sub_block_mid = tf.expand_dims( new_xyz,-2, name = 'sub_block_mid' )   # gpu_1/sa_layer0/sub_block_mid
        global_block_mid = tf.reduce_mean( sub_block_mid,1, keepdims=True, name = 'global_block_mid' )
        grouped_xyz_submid = grouped_xyz - sub_block_mid
        grouped_xyz_glomid = grouped_xyz - global_block_mid

        grouped_xyz_feed = []
        if 'raw' in self.xyz_elements:
            grouped_xyz_feed.append( grouped_xyz )
        if 'sub_mid' in self.xyz_elements:
            grouped_xyz_feed.append( grouped_xyz_submid )
        if 'global_mid' in self.xyz_elements:
            grouped_xyz_feed.append( grouped_xyz_glomid )
        grouped_xyz_feed = tf.concat( grouped_xyz_feed, -1 )

        if cascade_id==0:
            # xyz must be at the first in feed_data_elements !!!!
            grouped_points = tf.concat( [grouped_xyz_feed, grouped_points[...,3:]],-1 )

            #if len(indrop_keep_mask.get_shape()) != 0:
            #    if InDropMethod == 'set1st':
            #        # set all the dropped item as the first item
            #        tmp1 = tf.multiply( grouped_points, grouped_indrop_keep_mask )
            #        points_1st = grouped_points[:,:,0:1,:]
            #        points_1st = tf.tile( points_1st, [1,1,grouped_points.shape[2],1] )
            #        indrop_mask_inverse = 1 - grouped_indrop_keep_mask
            #        tmp2 = indrop_mask_inverse * points_1st
            #        grouped_points = tf.add( tmp1, tmp2, name='grouped_points_droped' ) # gpu_0/sa_layer0/grouped_points_droped
            #        #tf.add_to_collection( 'check', grouped_points )
            #    elif InDropMethod == 'set0':
            #        valid_mask = tf.logical_and( valid_mask, tf.equal(grouped_indrop_keep_mask,0), name='valid_mask_droped' )   # gpu_1/sa_layer0/valid_mask_droped

        elif use_xyz:
            grouped_points = tf.concat([grouped_xyz_feed, grouped_points],axis=-1)

        tf.add_to_collection( 'grouped_xyz', grouped_xyz )
        tf.add_to_collection( 'grouped_xyz_submid', grouped_xyz_submid )
        tf.add_to_collection( 'grouped_xyz_glomid', grouped_xyz_glomid )

        if cascade_id>0 and use_xyz and (not cascade_id==self.cascade_num-1):
            grouped_points = tf.concat([grouped_xyz_feed, grouped_points],axis=-1)

        nsample = grouped_points.get_shape()[2].value  # the conv kernel size

        if self.IsShowModel:
            print('\n\npointnet_sa_module cascade_id:%d\n xyz:%s\n grouped_xyz:%s\n new_xyz:%s\n grouped_points:%s\n nsample:%d'%(
                    cascade_id, tensor_info([xyz]), tensor_info([grouped_xyz]), tensor_info([new_xyz]), tensor_info([grouped_points]), nsample))

        new_points = grouped_points
        return new_xyz, grouped_xyz, new_points, valid_mask

  def sa_model(self, cascade_id, new_points, grouped_xyz, valid_mask, block_bottom_center_mm, scope, bn=True, tnet_spec=None):
    batch_size = new_points.get_shape()[0].value
    with tf.variable_scope(scope) as sc:
        if valid_mask!=None:
            new_points = new_points * tf.cast(valid_mask[:,:,:,0:1], tf.float32)
        if self.data_format == 'channels_first':
          assert False, "not ready yet"
          # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
          # This provides a large performance boost on GPU. See
          # https://www.tensorflow.org/performance/performance_guide#data_formats
          new_points = tf.transpose(new_points, [0, 3, 1, 2])

        for i, num_out_channel in enumerate(self.mlp_configs['point_encoder'][cascade_id]):
            new_points = conv2d_fixed_padding(
                inputs=new_points, filters=num_out_channel, kernel_size=1,
                strides=1, data_format=self.data_format)
            new_points = batch_norm(new_points, self.is_training, self.data_format)
            new_points = tf.nn.relu(new_points)

            if self.IsShowModel:
                print('point encoder1 %d, new_points:%s'%(i, tensor_info([new_points])))

        if cascade_id == 0:
            root_point_features = new_points
            #if InDropMethod == 'set0':
            #    if len(indrop_keep_mask.get_shape()) != 0:
            #            new_points = tf.identity(new_points,'points_before_droped') # gpu_0/sa_layer0/points_before_droped:0
            #            new_points = tf.multiply( new_points, grouped_indrop_keep_mask, name='droped_points' )   # gpu_0/sa_layer0/droped_points:0
        else:
            root_point_features = None

        pooling = self.mlp_configs['block_learning']
        if pooling == '3DCNN' and ( cascade_id == 0):
            pooling = 'max'

        #if pooling=='avg':
        #    new_points = tf_util.avg_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='avgpool1')
        #elif pooling=='weighted_avg':
        #    with tf.variable_scope('weighted_avg1'):
        #        dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
        #        exp_dists = tf.exp(-dists * 5)
        #        weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
        #        new_points *= weights # (batch_size, npoint, nsample, mlps_0[-1])
        #        new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
        if pooling=='max':
            # Even the grouped_points and grouped_xyz are 0 for invalid points, the
            # vaule after mlp will not be. It has to be set as 0 forcely before
            # pooling.
            if valid_mask!=None:
                new_points = new_points * tf.cast(valid_mask[:,:,:,0:1], tf.float32)
            new_points = tf.identity( new_points, 'points_before_max' )             # gpu_0/sa_layer0/points_before_max
            new_points = tf.reduce_max(new_points, axis=[2], keepdims=True, name='points_after_max')
        #elif pooling=='min':
        #    new_points = tf_util.max_pool2d(-1*new_points, [1,nsample], stride=[1,1], padding='VALID', scope='minpool1')
        #elif pooling=='max_and_avg':
        #    avg_points = tf_util.max_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='maxpool1')
        #    max_points = tf_util.avg_pool2d(new_points, [1,nsample], stride=[1,1], padding='VALID', scope='avgpool1')
        #    new_points = tf.concat([avg_points, max_points], axis=-1)
        elif pooling == '3DCNN':
            new_points = self.grouped_points_to_voxel_points( cascade_id, new_points, valid_mask, block_bottom_center_mm, grouped_xyz)
            if self.IsShowModel:
                print('voxel points:%s'%(tensor_info([new_points])))
            for i, num_out_channel in enumerate( self.mlp_configs['voxel_channels'][cascade_id] ):

                if self.mlp_configs['voxel_paddings'][cascade_id][i] == 0:
                  padding = 'VALID'
                else:
                  padding = 'SAME'
                if type(num_out_channel) == int:
                    new_points = conv3d_fixed_padding(
                            inputs = new_points,
                            filters = num_out_channel,
                            kernel_size = self.mlp_configs['voxel_kernels'][cascade_id][i],
                            strides = self.mlp_configs['voxel_strides'][cascade_id][i],
                            padding = padding,
                            data_format = self.data_format)
                    new_points = batch_norm(new_points, self.is_training, self.data_format)
                    new_points = tf.nn.relu(new_points)
                    if self.IsShowModel:
                        print('block learning by 3dcnn %d, new_points:%s'%(i, tensor_info([new_points])))
                elif num_out_channel == 'max' or 'ave':
                  if num_out_channel == 'max':
                    pool_fn = tf.layers.max_pooling3d
                  elif num_out_channel == 'ave':
                    pool_fn = tf.layers.average_pooling3d
                  new_points = pool_fn(
                              inputs = new_points,
                              pool_size = kernel_i,
                              strides = stride_i,
                              padding = 'valid',
                              name = '3d%s_%d'%(num_out_channel, i),
                              data_format = self.data_format)
                  if self.IsShowModel:
                      print('block learning m%s pooling %d, new_points:%s'%(num_out_channel, i, tensor_info([new_points])))
                # gpu_0/sa_layer4/3dconv_0/points_3dcnn_0:0
            if cascade_id < self.cascade_num-1:
              new_points = tf.squeeze( new_points, [1,2,3] )
            else:
              new_points = tf.squeeze( new_points )
            new_points = tf.reshape( new_points, [batch_size, -1, 1, new_points.shape[-1].value] )

        if self.IsShowModel:
            print('after %s, new_points:%s'%( pooling, tensor_info([new_points])))

        new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlps_1[-1])

        if self.IsShowModel:
            print('pointnet_sa_module return\n  new_points:%s\n\n'%(tensor_info([new_points])))
            #import pdb;pdb.set_trace()
        # (2, 512, 64)
        return new_points, root_point_features


  def grouped_points_to_voxel_points (self, cascade_id, new_points, valid_mask, block_bottom_center_mm, grouped_xyz):
    IS_merge_blocks_while_fix_bmap = True

    block_bottom_center_mm = tf.identity( block_bottom_center_mm,'block_bottom_center_mm' )      # gpu_0/sa_layer3/block_bottom_center_mm:0
    new_points = tf.identity(new_points,name='points_tov') # gpu_0/sa_layer4/points_tov:0
    c500 = tf.constant([500],tf.float32)
    c1000 = tf.constant([1000],tf.float32)
    c1 = tf.constant([1,1,1],tf.float32)
    step_last_org = self.sub_block_step_candis[cascade_id-1] * c1
    step_last = tf.minimum( step_last_org, self.max_step_stride, name='step_last' )    # gpu_0/sa_layer1/step_last:0
    step_last = tf.expand_dims(step_last,1)
    stride_last_org = self.sub_block_stride_candis[cascade_id-1] * c1
    stride_last = tf.minimum( stride_last_org, self.max_step_stride, name='stride_last' )  # gpu_0/sa_layer1/stride_last:0
    stride_last = tf.expand_dims(stride_last,1)

    voxel_bottom_xyz_mm = block_bottom_center_mm[:,:,0:3]
    # NOTE: c1=[1,1,1]*0.5 ONLY when the sh5 step is also the same on three dimensions.
    #                      Otherwise, the stride at each cascade may also be changed.
    min_point_bottom_xyz_mm = voxel_bottom_xyz_mm
    min_point_bottom_xyz_mm = tf.expand_dims( min_point_bottom_xyz_mm, -2, name='min_point_bottom_xyz_mm' ) # gpu_0/sa_layer1/min_point_bottom_xyz_mm:0
    grouped_bottom_xyz_mm = grouped_xyz * c1000 - step_last * c500  # gpu_0/sa_layer1/sub_1:0
        # For ExtraGlobal layer, the step_last may be cropped, thus the point_indices_f is smaller.
    point_indices_f = (grouped_bottom_xyz_mm - min_point_bottom_xyz_mm) / (stride_last*c1000)  # gpu_0/sa_layer3/div:0
    point_indices_f = tf.identity( point_indices_f, name='point_indices_f' )    # gpu_0/sa_layer4/point_indices_f:0

    # invalid indices comes from merge_blocks_while_fix_bmap
    # set point_indices_f for invalid points as
    # NETCONFIG['redundant_points_in_block'] ( shoud be set < -500)
    invalid_mask = tf.equal( valid_mask, False )
    invalid_mask = tf.tile( invalid_mask, [1,1,1,3], name='invalid_mask')  # gpu_0/sa_layer1/valid_mask:0
    point_indices_f = tf.where( invalid_mask, tf.ones(shape=point_indices_f.shape,dtype=tf.float32)*tf.constant( -9999,dtype=tf.float32), point_indices_f )
    point_indices = tf.rint( point_indices_f,'point_indices' )  # gpu_0/sa_layer3/point_indices:0
    point_indices_checkmin = tf.where( invalid_mask, tf.ones(shape=point_indices_f.shape,dtype=tf.float32)*tf.constant(999,dtype=tf.float32), point_indices, name='point_indices_checkmin' )

    # ------------------------------------------------------------------
    # check indice err
    Max_Assert_0 = 1e-4

    point_indices_err = tf.abs( point_indices - point_indices_f, name='point_indices_err' )     # gpu_0/sa_layer3/point_indices_err:0
    point_indices_maxerr = tf.reduce_max( point_indices_err, name='point_indices_maxerr_xyz' ) # gpu_0/sa_layer3/point_indices_maxerr_xyz:0
    check_point_indices = tf.assert_less( point_indices_maxerr, Max_Assert_0, data=[cascade_id, point_indices_maxerr],
                                            message='point indices in voxel check on cascade %d '%(cascade_id), name='check_point_indices' )
    tf.add_to_collection( 'check', check_point_indices )


    # check indice scope:
    # Actually only works when IS_merge_blocks_while_fix_bmap=False
    Max_Assert = 1e-4+5

    batch_size = new_points.shape[0].value
    block_num = new_points.shape[1].value
    point_num = new_points.shape[2].value
    channel_num = new_points.shape[3].value

    if self.dataset_name == 'MODELNET40':
        IsTolerateBug = 2
        IsTolerateBug = 0
    else:
        IsTolerateBug = 1

    if cascade_id==self.cascade_num-1:
        # only in this global cascde, the steps and strides in each dimension
        # can be different
        if self.dataset_name == 'MODELNET40' and self.global_step[0]==3.5:
            self.global_step = np.array( [2.3,2.3,2.3] )
        max_indice_f = ( np.abs(self.global_step) - np.array([1,1,1])*self.sub_block_step_candis[cascade_id-1] ) / (np.array([1,1,1])*self.sub_block_stride_candis[cascade_id-1])
        max_indice_v = np.rint( max_indice_f )
        if self.dataset_name != 'MODELNET40':
            assert np.sum(np.abs(max_indice_f-max_indice_v)) < Max_Assert
        max_indice_v += 1* IsTolerateBug

        voxel_size = max_indice_v.astype(np.int32)+1
        voxel_shape = [batch_size, block_num, voxel_size[0], voxel_size[1], voxel_size[2], channel_num]

        point_indices_checkmin = tf.identity(point_indices_checkmin, 'point_indices_checkmin_A') #
        point_indices_checkmin += (max_indice_v+2*IsTolerateBug) * IS_merge_blocks_while_fix_bmap
        point_indices_checkmin = tf.identity(point_indices_checkmin, 'point_indices_checkmin_B') # gpu_1/sa_layer4/point_indices_checkmin_B:0
        point_indices, first_unique_masks_global = unique_nd( point_indices )

        for i in range(3):
            real_max = tf.reduce_max(point_indices[:,:,:,i])
            check_max_indice = tf.assert_less( real_max - max_indice_v[i], tf.constant(Max_Assert + IS_merge_blocks_while_fix_bmap * max_indice_v[i], dtype=tf.float32 ),
                                              data=[cascade_id, i, real_max, max_indice_v[i]], name='check_max_indice_'+str(i) )
            tf.add_to_collection( 'check', check_max_indice )
        if self.IsShowModel:
            print( 'cascade:%d (global) \tvoxel size:%s'%(cascade_id, voxel_size) )

    else:
        max_indice_f = ( self.sub_block_step_candis[cascade_id] - self.sub_block_step_candis[cascade_id-1] ) / self.sub_block_stride_candis[cascade_id-1]
        max_indice_v = np.rint( max_indice_f ).astype(np.float32)
        assert abs(max_indice_f-max_indice_v) < Max_Assert + IS_merge_blocks_while_fix_bmap * max_indice_v
        voxel_size = max_indice_v.astype(np.int32)+1
        voxel_shape = [batch_size, block_num, voxel_size, voxel_size, voxel_size, channel_num]

        max_indice_1 = tf.constant(max_indice_v,tf.float32)
        real_max = tf.reduce_max(point_indices)
        check_max_indice = tf.assert_less( real_max - max_indice_1, tf.constant(Max_Assert + IS_merge_blocks_while_fix_bmap * max_indice_v, tf.float32 ),
                                          data=[cascade_id, real_max, max_indice_1], name='check_max_indice' )
        tf.add_to_collection( 'check', check_max_indice )
        point_indices_checkmin += (max_indice_v) * IS_merge_blocks_while_fix_bmap + IsTolerateBug*1
        if self.IsShowModel:
            print( 'cascade:%d \tvoxel size:%s'%(cascade_id, voxel_size) )


    point_indices_min = tf.reduce_min(point_indices_checkmin, name='point_indices_min') # gpu_0/sa_layer4/point_indices_min:0
    check_min_indice = tf.assert_less( tf.constant(-Max_Assert, tf.float32),
                                      point_indices_min, data=[cascade_id,point_indices_min], name='check_min_indice' )
    tf.add_to_collection( 'check', check_min_indice )
    # ------------------------------------------------------------------
    point_indices = tf.cast( point_indices, tf.int32, name='point_indices' )    # gpu_0/sa_layer1/point_indices_1:0
    batch_idx = tf.reshape( tf.range(batch_size),[batch_size,1,1,1] )
    batch_idx = tf.tile( batch_idx, [1,block_num,point_num,1] )
    bn_idx = tf.reshape( tf.range(block_num),[1,block_num,1,1] )
    bn_idx = tf.tile( bn_idx, [batch_size,1,point_num,1] )
    point_indices = tf.concat( [batch_idx, bn_idx, point_indices], -1, name='point_indices' ) # gpu_0/sa_layer4/point_indices_1:0

    # Note: if point_indices have replicated items, the responding value will be multiplied which will lead to error!
    # For global cascade, the replicated indices can come from replicated aim
    # block of the last gs cascade. This should be solved while generating point_indices for global in this function.
    # For other cascades, the replicated indices can come from replicated points
    #       inside aim block in bidxmap file. This shoule be solved by add np.unique  while merging blocks in bidxmap.
    voxel_points = tf.scatter_nd( point_indices, new_points, shape=voxel_shape, name='voxel_points' )   # gpu_0/sa_layer1/voxel_points:0

    # check voxel: takes long time, only perform for debug
    check_points = tf.gather_nd( voxel_points, point_indices, name='check_points' ) # gpu_0/sa_layer4/check_points:0
    scatter_err = tf.abs( check_points - new_points) # gpu_0/sa_layer1/scatter_err:0
    scatter_err = scatter_err * tf.cast(invalid_mask[:,:,:,0:1], tf.float32)
    scatter_err = tf.identity( scatter_err, name='scatter_err'  )
    scatter_err_max = tf.reduce_max( scatter_err, name = 'scatter_err_max') # gpu_0/sa_layer1/scatter_err_max:0
    points_check = tf.assert_less( scatter_err_max, Max_Assert, data=[cascade_id, scatter_err_max], name='scatter_check' )
    if DEBUG_TMP and not IS_merge_blocks_while_fix_bmap:
        tf.add_to_collection( 'check', points_check )

    # ------------------------------------------------------------------
    new_voxel_shape = tf.concat( [ tf.constant([batch_size*block_num],tf.int32), voxel_shape[2:6] ],0 )
    voxel_points = tf.reshape( voxel_points, shape = new_voxel_shape )
    #if self.aug_types['RotateVox']:
    #    voxel_points = rotate_voxel_randomly( voxel_points, configs )
    return voxel_points
