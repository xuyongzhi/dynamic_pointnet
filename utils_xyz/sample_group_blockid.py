# xyz Decc 2017
# Do 3d point cloud  sample and group by block index


def get_sample_group_idxs(npoint,block_step,nsample):

    return sample_idxs,group_idxs

def sample_and_group(npoint,block_step,nsample):
    '''
    Get npoint sub-blocks with equal stride and <block_step> step. The center of each sub-block is npoint down-sampled points.
    In each sub-block, nsample points are extracted.
    '''


    return new_xyz, sub_block_idxs,  group_idxs, grouped_xyz

