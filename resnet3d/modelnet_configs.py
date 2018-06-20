# xyz June 2018
import numpy as np

DEFAULTS = {}
DEFAULTS['residual'] = True
DEFAULTS['optimizer'] = 'adam'
DEFAULTS['learning_rate0'] = 0.001
DEFAULTS['batch_norm_decay'] = 0.9

DEFAULTS['model_flag'] = 'm'
DEFAULTS['resnet_size'] = 34
DEFAULTS['num_filters0'] = 32
DEFAULTS['feed_data'] = 'xyzs'
DEFAULTS['aug_types'] = 'r-360_0_0'
DEFAULTS['aug_types'] = ''
DEFAULTS['drop_imo'] = '0_0_5'
DEFAULTS['data_path'] = 'MODELNET40H5F/Merged_tfrecord/6_mgs1_gs2_2-mbf-neg_fmn14_mvp1-1024_240_1-64_27_256-0d2_0d4-0d1_0d2-pd3-2M2pp'
DEFAULTS['batch_size'] = 48
DEFAULTS['num_gpus'] = 2
DEFAULTS['train_epochs'] = 41
DEFAULTS['data_format'] = 'channels_last'

def get_block_paras(resnet_size, model_flag):
  block_sizes = {}
  block_kernels = {}
  block_strides = {}
  block_paddings = {}   # only used when strides == 1

  rs = 34
  block_sizes[rs]    = [[4], [3,1], [2,2,2]]
  block_kernels[rs]  = [[1], [2,3], [3,3,3]]
  block_strides[rs]  = [[1], [1,1], [1,1,1]]
  block_paddings[rs] = [['s'], ['s','v'], ['v','v','v']]

  rs = 50
  block_sizes[rs]    = [[5], [4,1], [3,3,3]]
  block_kernels[rs]  = [[1], [2,3], [3,3,3]]
  block_strides[rs]  = [[1], [1,1], [1,1,1]]
  block_paddings[rs] = [['s'], ['s','v'], ['v','v','v']]

  if 'V' not in model_flag:
    for i in range(len(block_sizes[resnet_size])):
      for j in range(len(block_sizes[resnet_size][i])):
        block_kernels[resnet_size][i][j] = 1
        block_strides[resnet_size][i][j] = 1
        block_paddings[resnet_size][i][j] = 'v'

  if resnet_size not in block_sizes:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, resnet_size.keys()))
    raise ValueError(err)

  # check settings
  for k in block_kernels:
    # cascade_id 0 is pointnet
    assert (np.array(block_kernels[k][0])==1).all()
    assert (np.array(block_strides[k][0])==1).all()

  return block_sizes, block_kernels, block_strides, block_paddings
