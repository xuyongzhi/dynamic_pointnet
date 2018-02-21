# xyz
# Jan 2017

from __future__ import print_function
import numpy as np


_gsbb_config = '2C1'
print('\n gsbb_config:%s \n-----------------------------------------------------'%(_gsbb_config))

def get_gsbb_config( config_flag = _gsbb_config ):
    flatbxmap_max_nearest_num = 1
    flatbxmap_max_dis = 4
    padding = 0.6
    if config_flag == '3A1':
        global_stride = np.array([1,1,-1]).astype(np.float)
        global_step = np.array([2.0,2.0,-1]).astype(np.float)
        max_global_num_point = 25600
        global_num_point = 25600

        sub_block_stride_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        sub_block_step_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        nsubblock_candis =       np.array([512,256, 64]).astype(np.int32)
        npoint_subblock_candis = np.array([128,  12,  6]).astype(np.int32)

    elif config_flag == '3B1':
        global_stride = np.array([1.6,1.6,-1]).astype(np.float)
        global_step = np.array([2.0,2.0,-1]).astype(np.float)
        max_global_num_point = 25600
        global_num_point = 25600

        sub_block_stride_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        sub_block_step_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        nsubblock_candis =       np.array([512,256, 64]).astype(np.int32)
        npoint_subblock_candis = np.array([128,  12,  6]).astype(np.int32)

    elif config_flag == '3B2':
        global_stride = np.array([1.6,1.6,-1]).astype(np.float)
        global_step = np.array([2.0,2.0,-1]).astype(np.float)
        max_global_num_point = 25600
        global_num_point = 25600

        sub_block_stride_candis = np.array([0.1,0.4,0.8]).astype(np.float)
        sub_block_step_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        nsubblock_candis =       np.array([1024,256, 64]).astype(np.int32)
        npoint_subblock_candis = np.array([192,  48,  6]).astype(np.int32)

    elif config_flag == '2C1':
        global_stride = np.array([0.5,0.5,-1]).astype(np.float)
        global_step = np.array([1,1,-1]).astype(np.float)
        max_global_num_point = 2048
        global_num_point = 2048

        sub_block_stride_candis = np.array([0.2,0.6]).astype(np.float)
        sub_block_step_candis = np.array([0.2,0.6]).astype(np.float)
        nsubblock_candis =       np.array([160,32]).astype(np.int32)
        npoint_subblock_candis = np.array([32,  12]).astype(np.int32)
    else:
        assert False,"gsbb config flag not recognized: %s"%(config_flag)
    return max_global_num_point, global_stride,global_step,global_num_point,sub_block_stride_candis,sub_block_step_candis,nsubblock_candis,\
            npoint_subblock_candis,config_flag, flatbxmap_max_nearest_num, flatbxmap_max_dis, padding



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
