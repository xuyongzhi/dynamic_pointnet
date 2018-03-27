# xyz
# Jan 2017

from __future__ import print_function
import numpy as np

#-------------------------------------------------------------------------------
NETCONFIG = {}
NETCONFIG['max_global_sample_rate'] = 5


#-------------------------------------------------------------------------------
# gsbb config
_gsbb_config = '3A1'
print('\n gsbb_config:%s \n-----------------------------------------------------'%(_gsbb_config))

def get_gsbb_config( gsbb_config = _gsbb_config ):
    flatbxmap_max_dis = 4
    padding = 0.6


    if gsbb_config == '3A1':
        # for scannet
        # _12800_1d6_2_fmn3-256_48_16-56_8_8-0d2_0d6_1d2-0d2_0d6_1d2
        global_stride = np.array([1.6,1.6,-1]).astype(np.float)
        global_step = np.array([2.0,2.0,-1]).astype(np.float)
        max_global_num_point = 12800
        global_num_point = 12800
        flatbxmap_max_nearest_num = 4

        sub_block_stride_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        sub_block_step_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        nsubblock_candis =       np.array([480, 80, 24]).astype(np.int32)
        npoint_subblock_candis = np.array([80,  20,  10]).astype(np.int32)

    elif gsbb_config == '3B2':
        global_stride = np.array([1.6,1.6,-1]).astype(np.float)
        global_step = np.array([2.0,2.0,-1]).astype(np.float)
        max_global_num_point = 12800
        global_num_point = 12800
        flatbxmap_max_nearest_num = 6

        sub_block_stride_candis = np.array([0.1,0.3,0.6]).astype(np.float)
        sub_block_step_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        nsubblock_candis =       np.array([2048, 256, 64]).astype(np.int32)
        npoint_subblock_candis = np.array([48,  32, 16]).astype(np.int32)


    elif gsbb_config == '2C1':
        global_stride = np.array([0.5,0.5,-1]).astype(np.float)
        global_step = np.array([1,1,-1]).astype(np.float)
        max_global_num_point = 2048
        global_num_point = 2048
        flatbxmap_max_nearest_num = 3

        sub_block_stride_candis = np.array([0.2,0.6]).astype(np.float)
        sub_block_step_candis = np.array([0.2,0.6]).astype(np.float)
        nsubblock_candis =       np.array([160,32]).astype(np.int32)
        npoint_subblock_candis = np.array([32,  12]).astype(np.int32)
    else:
        assert False,"gsbb config flag not recognized: %s"%(gsbb_config)
    return max_global_num_point, global_stride,global_step,global_num_point,sub_block_stride_candis,sub_block_step_candis,nsubblock_candis,\
            npoint_subblock_candis,gsbb_config, flatbxmap_max_nearest_num, flatbxmap_max_dis, padding

#-------------------------------------------------------------------------------
