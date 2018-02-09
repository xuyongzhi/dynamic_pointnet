# xyz
# Jan 2017

from __future__ import print_function
import numpy as np


_gsbb_config = '3B'
#_gsbb_config = '3C'
print('\n gsbb_config:%s \n-----------------------------------------------------'%(_gsbb_config))

def get_gsbb_config( config_flag = _gsbb_config ):
    max_global_num_point = 25600
    if config_flag == '3A':
        global_stride = np.array([1.6,1.6,-1]).astype(np.float)
        global_step = np.array([2.0,2.0,-1]).astype(np.float)
        global_num_point = 25600

        sub_block_stride_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        sub_block_step_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        nsubblock_candis =       np.array([512,256, 64]).astype(np.int32)
        npoint_subblock_candis = np.array([128,  12,  6]).astype(np.int32)

    elif config_flag == '3A1':
        global_stride = np.array([1.6,1.6,-1]).astype(np.float)
        global_step = np.array([2.0,2.0,-1]).astype(np.float)
        global_num_point = 8192

        sub_block_stride_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        sub_block_step_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        nsubblock_candis =       np.array([512,256, 64]).astype(np.int32)
        npoint_subblock_candis = np.array([128,  12,  6]).astype(np.int32)

    elif config_flag == '3B':
        global_stride = np.array([1.6,1.6,-1]).astype(np.float)
        global_step = np.array([2.0,2.0,-1]).astype(np.float)
        global_num_point = 25600

        sub_block_stride_candis = np.array([0.1,0.4,0.8]).astype(np.float)
        sub_block_step_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        nsubblock_candis =       np.array([2048,256, 64]).astype(np.int32)
        npoint_subblock_candis = np.array([384,  12,  6]).astype(np.int32)
    elif config_flag == '3C':
        global_stride = np.array([1.2,1.2,-1]).astype(np.float)
        global_step = np.array([2.0,2.0,-1]).astype(np.float)
        global_num_point = 25600

        sub_block_size_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        nsubblock_candis =       np.array([512,256, 64]).astype(np.int32)
        npoint_subblock_candis = np.array([128,  12,  6]).astype(np.int32)
    else:
        assert False,"gsbb config flag not recognized: %s"%(config_flag)
    return max_global_num_point, global_stride,global_step,global_num_point,sub_block_stride_candis,sub_block_step_candis,nsubblock_candis,npoint_subblock_candis,config_flag



#-------------------------------------------------------------------------------
'''
config_flag = '3C'
bidxmaps_sample_group   shape= (56, 320, 12)
global_stride:
[ 1.2, 1.2,-1. ]
global_step:
[ 2., 2.,-1.]
sub_block_size_candis:
[0.2,0.6,1.2]
nsubblock_candis:
[512,256, 64]
npoint_subblock_candis:
[128, 12,  6]
sum_flatten_bmap_sample_num:
['flatten_fixed_num', 'flatten_valid_num', 'block_num']
global_block_num: 56
[[25600.  ,24361.5 ,    1.  ],
 [  512.  ,  315.18,    1.  ],
 [  256.  ,   36.41,    1.  ]]
valid_num:
56
sum_sg_bidxmap_sample_num:
['missed_baseb_num', 'all_baseb_num', 'nsubblock', 'valid_subblock_num', 'unit_block_num', 'npoint_subblock', 'valid_npoint_subblock', 'subblock_num']
global_block_num: 56	subblock_num: [17650.  2039.   435.]
[[   84.91,31006.8 ,  512.  ,  316.52,    1.  ,  128.  ,   98.11,    1.  ],
 [    0.  ,  315.18,  256.  ,  155.79,    1.  ,   12.  ,    8.66,    1.  ],
 [    0.  ,   36.41,   64.  ,   32.84,    1.  ,    6.  ,    4.69,    1.  ]]
'''
