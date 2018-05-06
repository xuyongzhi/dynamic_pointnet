
from __future__ import print_function
import os
import sys
import numpy as np
import h5py
import glob
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
UPER_DIR = os.path.dirname(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR,'data')

class GLOBAL_PARA():
    '''
    outout:  data_path[data_source][folder]
    '''
    data_source_ls = ['stanford_indoor3d','scannet_data','ETH']
    dataset_path = {}
    dataset_dir = {}
    for data_source in data_source_ls:
        ds_dir  = os.path.join(DATA_DIR,data_source)
        dataset_path[data_source] = {}
        dataset_dir[data_source] = ds_dir

#        folders = ['raw','rawh5','stride_0d5_step_0d5','stride_0d5_step_1',
#                   'stride_0d5_step_1_4096','stride_0d5_step_1_4096_norm']
#        for folder in folders:
#            dataset_path[data_source][folder] = os.path.join(ds_dir,folder)


    h5_num_row_1M = 50*1000
    h5_num_row_10M = h5_num_row_1M * 10
    h5_num_row_100M = h5_num_row_1M * 100
    h5_num_row_1G = h5_num_row_1M * 1024
    h5_chunk_row_step =  h5_num_row_1M

    @classmethod
    def sample(cls,org_N,sample_N,sample_method='random'):
        if sample_method == 'random':
            if org_N == sample_N:
                sample_choice = np.arange(sample_N)
            elif org_N > sample_N:
                sample_choice = np.random.choice(org_N,sample_N)
                #reduced_num += org_N - sample_N
            else:
                #sample_choice = np.arange(org_N)
                new_samp = np.random.choice(org_N,sample_N-org_N)
                sample_choice = np.concatenate( (np.arange(org_N),new_samp) )
            #str = '%d -> %d  %d%%'%(org_N,sample_N,100.0*sample_N/org_N)
            #print(str)
        return sample_choice


if __name__ == "__main__":
    print( GLOBAL_PARA.dataset_dir['stanford_indoor3d']  )
    print( GLOBAL_PARA.dataset_dir['stanford_indoor3d']['raw']  )
    print( GLOBAL_PARA.dataset_dir['stanford_indoor3d']['rawh5']  )
