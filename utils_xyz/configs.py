# xyz
# Jan 2017

from __future__ import print_function
import numpy as np

#-------------------------------------------------------------------------------
NETCONFIG = {}
NETCONFIG['max_global_sample_rate'] = 3   # sample_res_num / org_num  This is very large for global block with few points which should be abandoned.
NETCONFIG['merge_blocks_while_fix_bmap'] = True
NETCONFIG['redundant_points_in_block'] = -7777777  # 'replicate' or a negative number to be asigned in bidxmap  (<-500)
#-------------------------------------------------------------------------------
# gsbb config
_gsbb_config = '4A2'
print('\n gsbb_config:%s \n-----------------------------------------------------'%(_gsbb_config))

def get_gsbb_config( gsbb_config = _gsbb_config ):
    '''
    global_step: When this <0, it's set to whole scene scope. But the limit is -global_step.
    global_stride: When global_step <0, should also <0
    '''

    flatbxmap_max_dis = 4
    padding = 0.3


    if gsbb_config == 'benz_d3':
         ## generating the KITTI dataset benchmark
        global_stride = np.array([-100,-100,-100]).astype(np.float)
        global_step = np.array([-100,-100,-100]).astype(np.float)
        max_global_num_point = 32768
        global_num_point = 32768
        flatbxmap_max_nearest_num = [-10, -10, -10]  # do not generate flatbxmap
        NETCONFIG['max_global_sample_rate'] = 5
        NETCONFIG['merge_blocks_while_fix_bmap'] = False


        sub_block_stride_candis = np.array([0.2, 0.4, 0.4]).astype(np.float)
        sub_block_step_candis   = np.array([0.4, 0.8, 1.8]).astype(np.float)
        nsubblock_candis        =np.array([12800, 4800, 2400]).astype(np.int32)
        npoint_subblock_candis = np.array([16,  8, 8]).astype(np.int32)


    #---------------------------------------------------------------------------
    elif gsbb_config == '3A1':
        # for scannet
        # _12800_1d6_2_fmn3-256_48_16-56_8_8-0d2_0d6_1d2-0d2_0d6_1d2
        global_stride = np.array([1.6,1.6,-8]).astype(np.float)
        global_step = np.array([2.0,2.0,-8]).astype(np.float)
        global_num_point = 12800
        flatbxmap_max_nearest_num = 4

        sub_block_stride_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        sub_block_step_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        nsubblock_candis =       np.array([480, 80, 24]).astype(np.int32)
        npoint_subblock_candis = np.array([80,  20,  10]).astype(np.int32)

    elif gsbb_config == '3B1':
        global_stride = np.array([-6,-6,-6]).astype(np.float)
        global_step = np.array([-10,-10,-6]).astype(np.float)
        global_num_point = 320000
        flatbxmap_max_nearest_num = 4

        sub_block_stride_candis = np.array([0.2,0.8,2]).astype(np.float)
        sub_block_step_candis = np.array([0.2,0.8,2]).astype(np.float)
        nsubblock_candis =       np.array([6400, 640, 128]).astype(np.int32)
        npoint_subblock_candis = np.array([80,  20,  16]).astype(np.int32)

    elif gsbb_config == '3B2':
        global_stride = np.array([-6,-6,-6]).astype(np.float)
        global_step = np.array([-10,-10,-10]).astype(np.float)
        global_num_point = 320000
        flatbxmap_max_nearest_num = 4

        sub_block_stride_candis = np.array([0.2,0.6,1.2]).astype(np.float)
        sub_block_step_candis = np.array([0.3,0.9,2.4]).astype(np.float)
        nsubblock_candis =       np.array([6400, 640, 128]).astype(np.int32)
        npoint_subblock_candis = np.array([200,  32,  48]).astype(np.int32)

    elif gsbb_config == '3B3':
        global_stride = np.array([-6,-6,-6]).astype(np.float)
        global_step = np.array([-10,-10,-10]).astype(np.float)
        global_num_point = 320000

        flatbxmap_max_nearest_num = 4
        sub_block_stride_candis = np.array([0.1,0.2,0.6,1.2]).astype(np.float)
        sub_block_step_candis = np.array([0.1,0.4,1.0,2.4]).astype(np.float)
        nsubblock_candis =       np.array([8000, 4800, 320, 56]).astype(np.int32)
        npoint_subblock_candis = np.array([100, 20,  40,  32]).astype(np.int32)

    elif gsbb_config == '3B4':
        global_stride = np.array([-6,-6,-6]).astype(np.float)
        global_step = np.array([-10,-10,-10]).astype(np.float)
        global_num_point = 128000
        flatbxmap_max_nearest_num = 4

        sub_block_stride_candis = np.array([0.1,0.2,0.6,1.2]).astype(np.float)
        sub_block_step_candis = np.array([0.1,0.4,1.0,2.4]).astype(np.float)
        nsubblock_candis =       np.array([8000, 4800, 320, 64]).astype(np.int32)
        npoint_subblock_candis = np.array([24, 20,  40,  32]).astype(np.int32)


    elif gsbb_config == '3C2':
        global_stride = np.array([-3,-3,-10]).astype(np.float)
        global_step = np.array([-4.8,-4.8,-10]).astype(np.float)
        global_num_point = 60000
        flatbxmap_max_nearest_num = 6

        sub_block_stride_candis = np.array([0.2,0.4,1.2]).astype(np.float)
        sub_block_step_candis =   np.array([0.2,0.6,1.8]).astype(np.float)
        nsubblock_candis =       np.array([1600, 480, 48]).astype(np.int32)
        npoint_subblock_candis = np.array([80,  16,  32]).astype(np.int32)

    elif gsbb_config == '4A1':  # ***
        global_stride = np.array([-3.6,-3.6,-1.8]).astype(np.float)
        global_step = np.array([-6.3,-6.3,-4.5]).astype(np.float)
        global_num_point = 10000 * 9
        flatbxmap_max_nearest_num = [1,4,4,4]

        sub_block_stride_candis = np.array([0.1,0.2,0.6,1.8]).astype(np.float)
        sub_block_step_candis   = np.array([0.1,0.3,0.9,2.7]).astype(np.float)
        nsubblock_candis =       np.array([6400, 2400, 320, 32]).astype(np.int32)
        npoint_subblock_candis = np.array([32, 16,  32,  48]).astype(np.int32)

    elif gsbb_config == '4A2':  # ***
        global_stride = np.array([-3.6,-3.6,-1.8]).astype(np.float)
        global_step = np.array([-6.3,-6.3,-4.5]).astype(np.float)
        global_num_point = 10000 * 9
        flatbxmap_max_nearest_num = [1,4,4,4]

        sub_block_stride_candis = np.array([0.1,0.2,0.6,1.8]).astype(np.float)
        sub_block_step_candis   = np.array([0.1,0.3,0.9,2.7]).astype(np.float)
        nsubblock_candis =       np.array([6400, 2400, 320, 48]).astype(np.int32)
        npoint_subblock_candis = np.array([32, 27,  64,  64]).astype(np.int32)

    else:
        assert False,"gsbb config flag not recognized: %s"%(gsbb_config)

    gsbb_config_dic = {}
    gsbb_config_dic['max_global_num_point'] = global_num_point
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
