# xyz
# Jan 2017

from __future__ import print_function
import numpy as np


config_flag = '3A'

def get_gsbb_config( config_flag ):
    if config_flag == '3A':
        global_stride = np.array([1.0,1.0,-1]).astype(np.float)
        global_step = np.array([2.0,2.0,-1]).astype(np.float)
        global_num_point = 20480

        sub_block_size_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        nsubblock_candis =       np.array([512,256, 64]).astype(np.int32)
        npoint_subblock_candis = np.array([128,  12,  6]).astype(np.int32)
    else:
        assert False,"gsbb config flag not recognized: %s"%(config_flag)
    return global_stride,global_step,global_num_point,sub_block_size_candis,nsubblock_candis,npoint_subblock_candis


