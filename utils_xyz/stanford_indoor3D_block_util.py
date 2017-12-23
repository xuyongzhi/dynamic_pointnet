# xyz
import os
import sys
import time
import numpy as np
import glob
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from block_data_prep_util import  Raw_H5f,Sort_RawH5f,Sorted_H5f

ROOT_DIR = os.path.dirname(BASE_DIR)
UPER_DIR = os.path.dirname(ROOT_DIR)
DATA_DIR = os.path.join(ROOT_DIR,'data')
STANFORD_DIR = os.path.join(DATA_DIR,'stanford_indoor3d')

START_T = time.time()
data_source = 'stanford_indoor3d'
class Indoor3d_Prepare():
    '''
    source:  from "http://buildingparser.stanford.edu/dataset.html"
    Tihe work flow of processing stanford_indoor3d data:
        (1) collectto each room to format: [x y z r g b label]. Done by Qi's code
        (2) gen each room to Raw_H5f
        (3) sort each room to Sorted_H5f with step = stride = 0.5
        (4) merge each room to Sorted_H5f with step = 1 & stride = 0.5
        (5) sample each block to NUM_POINT points and normalize each block
    '''
    raw_npy_path = os.path.join(STANFORD_DIR,'raw')
    rawh5f_path = os.path.join(STANFORD_DIR,'rawh5')
    sorted_path_stride_0d5_step_0d5 = os.path.join(STANFORD_DIR,'stride_0d5_step_0d5')
    sorted_path_stride_1_step_1 = os.path.join(STANFORD_DIR,'stride_1_step_1')
    sorted_path_stride_1_step_2 = os.path.join(STANFORD_DIR,'stride_1_step_2')

    @staticmethod
    def gen_stanford_indoor3d_to_rawh5f():
        file_list = glob.glob( os.path.join( Indoor3d_Prepare.raw_npy_path,'*.npy' ) )
        print('%d files in \n%s'%(len(file_list),Indoor3d_Prepare.raw_npy_path))
        rawh5f_path = Indoor3d_Prepare.rawh5f_path
        if not os.path.exists(rawh5f_path):
            os.makedirs(rawh5f_path)
        for fn in file_list:
            base_name = os.path.splitext(os.path.basename(fn))[0]
            h5_fn = os.path.join(rawh5f_path,base_name+'.rh5')
            with h5py.File(h5_fn,'w') as h5f:
                raw_h5f = Raw_H5f(h5f,h5_fn,'STANFORD_INDOOR3D')
                data = np.load(fn)
                num_row = data.shape[0]
                raw_h5f.set_num_default_row(num_row)
                raw_h5f.append_to_dset('xyz',data[:,0:3])
                raw_h5f.append_to_dset('color',data[:,3:6])
                raw_h5f.append_to_dset('label',data[:,6:7])
                raw_h5f.create_done()

    @staticmethod
    def SortRaw():
        file_list = glob.glob( os.path.join(Indoor3d_Prepare.rawh5f_path,'*.rh5') )
        block_step_xyz = [0.5,0.5,0.5]
        print('%d files in %s'%(len(file_list),Indoor3d_Prepare.raw_npy_path))
        sorted_path = Indoor3d_Prepare.sorted_path_stride_0d5_step_0d5
        Sort_RawH5f(file_list,block_step_xyz,sorted_path)

    @staticmethod
    def MergeSampleNorm():
        '''
         1 merge to new block step/stride size
             obj_merged: generate obj for merged
         2 randomly sampling to fix point number in each block
             obj_sampled_merged
         3 normalizing sampled block
        '''
        file_list = glob.glob( os.path.join(Indoor3d_Prepare.sorted_path_stride_0d5_step_0d5, \
                    '*.sh5') )
        new_stride = [1,1,-1]
        new_step = [2,2,-1]
        more_actions_config = {}
        more_actions_config['actions'] = ['merge','sample_merged','norm_sampled_merged']
        #more_actions_config['actions'] = ['merge','obj_merged','sample_merged','obj_sampled_merged']
        more_actions_config['sample_num'] = 4096
        for fn in file_list:
            with h5py.File(fn,'r') as f:
                sorted_h5f = Sorted_H5f(f,fn)
                sorted_h5f.merge_to_new_step(new_stride,new_step,
                        Indoor3d_Prepare.sorted_path_stride_1_step_2,more_actions_config)

    @staticmethod
    def SampleNorm():
        '''
        sample to fix point number
        normolize sampled block
        gen obj for sampled
        '''
        file_list = glob.glob( os.path.join(Indoor3d_Prepare.sorted_path_stride_1_step_2, \
                    '*.sh5') )
        sample_num = 8192
        gen_norm = True
        gen_obj = False
        for fn in file_list:
            with h5py.File(fn,'r') as f:
                sorted_h5f = Sorted_H5f(f,fn)
                sorted_h5f.file_random_sampling(sample_num,gen_norm,gen_obj)

    # Not used now
    @staticmethod
    def CombineAreaRooms_Normedh5():
        # and add area num dataset
        # combine normedh5 files together
        for area_no in range(1,7):
            area_str = 'Area_'+str(area_no)
            path = Indoor3d_Prepare.sorted_path_stride_1_step_2_8192_normed
            file_list = glob.glob( os.path.join(path, \
                                    area_str+'*.nh5' ) )
            print('file num = %d'%(len(file_list)))
            postfix = os.path.basename(file_list[0]).split('_stride_')[1]
            root_path = os.path.dirname(path)
            merged_fn = os.path.join(root_path,area_str+'_stride_'+postfix)
            print('merged file name: %s'%(merged_fn))
            with h5py.File(merged_fn,'w') as f:
                merged_normed_h5f = Normed_H5f(f,merged_fn)
                for k,fn in enumerate(file_list):
                    if k==0:
                        with h5py.File(fn,'r') as f0:
                            normed_h5f_0 = Normed_H5f(f0,fn)
                            data_shape = normed_h5f_0.get_data_shape()
                            merged_normed_h5f.create_dsets(0,data_shape[1],data_shape[2])
                            merged_normed_h5f.create_areano_dset(0,data_shape[1])
                    merged_normed_h5f.merge_file(fn)

    @staticmethod
    def Norm():
        file_list = glob.glob( os.path.join(GLOBAL_PARA.stanford_indoor3d_stride_0d5_step_1_4096,\
                              '*.sh5') )
        for fn in file_list:
            with h5py.File(fn,'r') as f:
                sorted_h5f = Sorted_H5f(f,fn)
                sorted_h5f.file_normalization()


if __name__ == '__main__':
    #Indoor3d_Prepare.gen_stanford_indoor3d_to_rawh5f()
    #Indoor3d_Prepare.SortRaw()
    Indoor3d_Prepare.MergeSampleNorm()
    #Indoor3d_Prepare.SampleNorm()
    #Indoor3d_Prepare.Norm()
    T = time.time() - START_T
    print('exit main, T = ',T)
