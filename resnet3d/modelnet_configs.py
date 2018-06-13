# xyz June 2018
import numpy as np

def get_block_paras(resnet_size):
  block_sizes = {}
  block_kernels = {}
  block_strides = {}
  block_paddings = {}   # only used when strides == 1

  rs = 34
  block_sizes[rs]    = [[3], [3,1], [2,2,2]]
  block_kernels[rs]  = [[1], [2,3], [3,3,3]]
  block_strides[rs]  = [[1], [1,1], [1,1,1]]
  block_paddings[rs] = [['s'], ['s','v'], ['v','v','v']]

  rs = 50
  block_sizes[rs]    = [[3], [3,1], [2,2,2]]
  block_kernels[rs]  = [[1], [2,3], [3,3,3]]
  block_strides[rs]  = [[1], [1,1], [1,1,1]]
  block_paddings[rs] = [['s'], ['s','v'], ['v','v','v']]

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
