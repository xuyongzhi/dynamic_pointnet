# xyz
# Jan 2017

from __future__ import print_function
import numpy as np

#-------------------------------------------------------------------------------
NETCONFIG = {}
NETCONFIG['max_global_sample_rate'] = 3   # sample_res_num / org_num

#-------------------------------------------------------------------------------
# gsbb config
_gsbb_config = '3B3'
print('\n gsbb_config:%s \n-----------------------------------------------------'%(_gsbb_config))

def get_gsbb_config( gsbb_config = _gsbb_config ):
    '''
    global_step: When this <0, it's set to whole scene scope. But the limit is -global_step.
    global_stride: When global_step <0, should also <0
    '''

    flatbxmap_max_dis = 4
    padding = 0.6

    if gsbb_config == '3A1':
        # for scannet
        # _12800_1d6_2_fmn3-256_48_16-56_8_8-0d2_0d6_1d2-0d2_0d6_1d2
        global_stride = np.array([1.6,1.6,-8]).astype(np.float)
        global_step = np.array([2.0,2.0,-8]).astype(np.float)
        max_global_num_point = 12800
        global_num_point = 12800
        flatbxmap_max_nearest_num = 4

        sub_block_stride_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        sub_block_step_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        nsubblock_candis =       np.array([480, 80, 24]).astype(np.int32)
        npoint_subblock_candis = np.array([80,  20,  10]).astype(np.int32)

    elif gsbb_config == '3B1':
        # for scannet
        # _12800_1d6_2_fmn3-256_48_16-56_8_8-0d2_0d6_1d2-0d2_0d6_1d2
        global_stride = np.array([-6,-6,-6]).astype(np.float)
        global_step = np.array([-10,-10,-10]).astype(np.float)
        max_global_num_point = 128000
        global_num_point = 128000
        flatbxmap_max_nearest_num = 4

        sub_block_stride_candis = np.array([0.2,0.8,2]).astype(np.float)
        sub_block_step_candis = np.array([0.2,0.8,2]).astype(np.float)
        nsubblock_candis =       np.array([6400, 640, 128]).astype(np.int32)
        npoint_subblock_candis = np.array([80,  20,  16]).astype(np.int32)

    elif gsbb_config == '3B2':
        # for scannet
        # _12800_1d6_2_fmn3-256_48_16-56_8_8-0d2_0d6_1d2-0d2_0d6_1d2
        global_stride = np.array([-6,-6,-6]).astype(np.float)
        global_step = np.array([-10,-10,-10]).astype(np.float)
        max_global_num_point = 320000
        global_num_point = 320000
        flatbxmap_max_nearest_num = 4

        sub_block_stride_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        sub_block_step_candis = np.array([0.3,0.9,2.4]).astype(np.float)
        nsubblock_candis =       np.array([6400, 640, 128]).astype(np.int32)
        npoint_subblock_candis = np.array([200,  32,  48]).astype(np.int32)

    elif gsbb_config == '3B3':
        # for scannet
        global_stride = np.array([-6,-6,-6]).astype(np.float)
        global_step = np.array([-10,-10,-10]).astype(np.float)
        max_global_num_point = 320000
        global_num_point = 320000
        flatbxmap_max_nearest_num = 4

        sub_block_stride_candis = np.array([0.1,0.2,0.6,1.2]).astype(np.float)
        sub_block_step_candis = np.array([0.1,0.4,1.0,2.4]).astype(np.float)
        nsubblock_candis =       np.array([8000, 4800, 320, 56]).astype(np.int32)
        npoint_subblock_candis = np.array([100, 20,  40,  32]).astype(np.int32)


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

    gsbb_config_dic = {}
    gsbb_config_dic['max_global_num_point'] = max_global_num_point
    gsbb_config_dic['global_stride'] = global_stride
    gsbb_config_dic['global_step'] = global_step
    gsbb_config_dic['global_num_point'] = global_num_point
    gsbb_config_dic['sub_block_stride_candis'] = sub_block_stride_candis
    gsbb_config_dic['sub_block_step_candis'] = sub_block_step_candis
    gsbb_config_dic['nsubblock_candis'] = nsubblock_candis
    gsbb_config_dic['npoint_subblock_candis'] = npoint_subblock_candis
    gsbb_config_dic['gsbb_config'] = gsbb_config
    gsbb_config_dic['flatbxmap_max_nearest_num'] = flatbxmap_max_nearest_num
    gsbb_config_dic['flatbxmap_max_dis'] = flatbxmap_max_dis
    gsbb_config_dic['padding'] = padding

    return  gsbb_config_dic

#-------------------------------------------------------------------------------
