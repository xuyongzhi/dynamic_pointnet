# xyz
from __future__ import print_function
import pdb, traceback
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from block_data_prep_util import Raw_H5f, Sort_RawH5f,Sorted_H5f
import numpy as np
import h5py
import glob
import time
import multiprocessing as mp
import itertools
import pickle


ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR,'data')
DATA_SOURCE= 'scannet_data'
SCANNET_DATA_DIR = os.path.join(DATA_DIR,DATA_SOURCE)

class Scannet_Prepare():
    '''

    '''
    rawh5f_dir_base =  os.path.join(SCANNET_DATA_DIR,'rawh5')
    sorted_path_stride_0d5_step_0d5 = os.path.join(SCANNET_DATA_DIR,'stride_0d5_step_0d5')
    sorted_path_stride_1_step_2 = os.path.join(SCANNET_DATA_DIR,'stride_1_step_2')
    sorted_path_stride_2_step_4 = os.path.join(SCANNET_DATA_DIR,'stride_2_step_4')

    @staticmethod
    def Load_Raw_Scannet_Pickle(split='test'):
        file_name = os.path.join(SCANNET_DATA_DIR,'scannet_%s.pickle'%(split))
        rawh5f_dir = Scannet_Prepare.rawh5f_dir_base +'_'+ split
        if not os.path.exists(rawh5f_dir):
            os.makedirs(rawh5f_dir)

        with open(file_name,'rb') as fp:
            scene_points_list = pickle.load(fp)
            semantic_labels_list = pickle.load(fp)

            print('%d scans for file:\n %s'%(len(semantic_labels_list),file_name))
            for n in range(len(semantic_labels_list)):
                # write one RawH5f file for one scane
                rawh5f_fn = os.path.join(rawh5f_dir,'scan_%d.rh5'%(n))
                num_points = semantic_labels_list[n].shape[0]
                with h5py.File(rawh5f_fn,'w') as h5f:
                    raw_h5f = Raw_H5f(h5f,rawh5f_fn,'SCANNET')
                    raw_h5f.set_num_default_row(num_points)
                    raw_h5f.append_to_dset('xyz',scene_points_list[n])
                    raw_h5f.append_to_dset('label',semantic_labels_list[n])
                    raw_h5f.create_done()
    @staticmethod
    def SortRaw(split='test'):
        rawh5f_dir = Scannet_Prepare.rawh5f_dir_base +'_'+ split
        file_list = glob.glob( os.path.join(rawh5f_dir,'*.rh5') )
        block_step_xyz = [0.5,0.5,0.5]
        print("%s files in %s"%(len(file_list),rawh5f_dir))
        sorted_path = Scannet_Prepare.sorted_path_stride_0d5_step_0d5+'_'+split
        Sort_RawH5f(file_list,block_step_xyz,sorted_path)

    @staticmethod
    def MergeSampleNorm(split):
        '''
         1 merge to new block step/stride size
             obj_merged: generate obj for merged
         2 randomly sampling to fix point number in each block
             obj_sampled_merged
         3 normalizing sampled block
        '''
        sorted_path = Scannet_Prepare.sorted_path_stride_0d5_step_0d5+'_'+split
        sorted_path_new = Scannet_Prepare.sorted_path_stride_1_step_2+'_'+split

        file_list = glob.glob( os.path.join(sorted_path, \
                    '*.sh5') )
        print('%d sh5 files in %s'%(len(file_list),sorted_path))
        new_stride = [1,1,-1]
        new_step = [2,2,-1]
        more_actions_config = {}
        #more_actions_config['actions'] = []
        #more_actions_config['actions'] = ['sample_merged']
        more_actions_config['actions'] = ['merge','sample_merged','norm_sampled_merged']
        more_actions_config['sample_num'] = 4096
        for fn in file_list:
            with h5py.File(fn,'r') as f:
                sorted_h5f = Sorted_H5f(f,fn)
                sorted_h5f.merge_to_new_step(new_stride,new_step,\
                        sorted_path_new,more_actions_config)

    @staticmethod
    def SampleNorm():
        sorted_path = Scannet_Prepare.sorted_path_stride_1_step_2+'_'+split
        file_list = glob.glob( os.path.join(sorted_path, '*.sh5' ))
        sample_num = 8192
        gen_norm = True
        gen_obj = False

        for fn in file_list:
            with h5py.File(fn,'r') as f:
                sorted_h5f = Sorted_H5f(f,fn)
                sorted_h5f.file_random_sampling(sample_num,gen_norm,gen_obj)


if __name__ == '__main__':
    #try:
        split = 'test_small'
        #Scannet_Prepare.Load_Raw_Scannet_Pickle(split)
        Scannet_Prepare.SortRaw(split)
        #Scannet_Prepare.MergeSampleNorm(split)
        #Scannet_Prepare.SampleNorm()
#    except:
#        type, value, tb = sys.exc_info()
#        traceback.print_exc()
#        pdb.post_mortem(tb)

