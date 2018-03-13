#xyz
from __future__ import print_function
import pdb, traceback
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from block_data_prep_util import Raw_H5f, Sort_RawH5f,Sorted_H5f,Normed_H5f,show_h5f_summary_info,MergeNormed_H5f,get_stride_step_name
from block_data_prep_util import GlobalSubBaseBLOCK,get_mean_sg_sample_rate,get_mean_flatten_sample_rate,check_h5fs_intact
import numpy as np
import h5py
import glob
import time
import multiprocessing as mp
import itertools
import pickle

TMPDEBUG = True
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR,'data')
DATA_SOURCE= 'Scannet_H5F'
SCANNET_DATA_DIR = os.path.join(DATA_DIR,DATA_SOURCE)



def WriteSortH5f_FromRawH5f(rawh5_file_ls,block_step_xyz,sorted_path,IsShowInfoFinished):
    Sort_RawH5f(rawh5_file_ls,block_step_xyz,sorted_path,IsShowInfoFinished)
    return rawh5_file_ls

def GenPyramidSortedFlie(fn):
    with h5py.File(fn,'r') as f:
        sorted_h5f = Sorted_H5f(f,fn)
        Always_CreateNew_pyh5 = False
        Always_CreateNew_bmh5 = False
        Always_CreateNew_bxmh5 = False
        if TMPDEBUG:
            Always_CreateNew_bmh5 = True
            Always_CreateNew_pyh5 = True
            Always_CreateNew_bxmh5 = True

        sorted_h5f.file_saveas_pyramid_feed(True,Always_CreateNew_pyh5 = Always_CreateNew_pyh5, Always_CreateNew_bmh5 = Always_CreateNew_bmh5, Always_CreateNew_bxmh5=Always_CreateNew_bxmh5 )
    return fn

class Scannet_Prepare():
    '''

    '''
    scans_h5f_dir = os.path.join( SCANNET_DATA_DIR,'scans' )

    def __init__(self,split='test'):
        self.split = split
        self.rawh5f_dir =  self.scans_h5f_dir+'/rawh5/%s'%(split)

        self.sorted_path_stride_0d5_step_0d5 = os.path.join(SCANNET_DATA_DIR,'stride_0d5_step_0d5')+'_'+split
        self.sorted_path_stride_1_step_2 = os.path.join(SCANNET_DATA_DIR,'stride_1_step_2')+'_'+split
        self.sorted_path_stride_1_step_2_8192 = os.path.join(SCANNET_DATA_DIR,'stride_1_step_2')+'_'+split+'_8192'
        self.sorted_path_stride_1_step_2_8192_norm = os.path.join(SCANNET_DATA_DIR,'stride_1_step_2')+'_'+split+'_8192_normed'
        self.filename_stride_1_step_2_8192_norm_merged = os.path.join(SCANNET_DATA_DIR,'stride_1_step_2')+'_'+split+'_8192_normed.nh5'
        self.sorted_path_stride_2_step_4 = os.path.join(SCANNET_DATA_DIR,'stride_2_step_4')+'_'+split
        self.sorted_path_stride_2_step_4_8192 = os.path.join(SCANNET_DATA_DIR,'stride_2_step_4')+'_'+split+'_8192'
        self.sorted_path_stride_2_step_4_8192_norm = os.path.join(SCANNET_DATA_DIR,'stride_2_step_4')+'_'+split+'_8192_normed'
        self.filename_stride_2_step_4_8192_norm_merged = os.path.join(SCANNET_DATA_DIR,'stride_2_step_4')+'_'+split+'_8192_normed.nh5'

    def Load_Raw_Scannet_Pickle(self):
        file_name = os.path.join(SCANNET_DATA_DIR,'scannet_%s.pickle'%(self.split))
        rawh5f_dir = self.rawh5f_dir
        if not os.path.exists(rawh5f_dir):
            os.makedirs(rawh5f_dir)

        with open(file_name,'rb') as fp:
            scene_points_list = pickle.load(fp)
            semantic_labels_list = pickle.load(fp)

            print('%d scans for file:\n %s'%(len(semantic_labels_list),file_name))
            for n in range(len(semantic_labels_list)):
                if TMPDEBUG and n>0: break
                # write one RawH5f file for one scane
                rawh5f_fn = os.path.join(rawh5f_dir,self.split+'_%d.rh5'%(n))
                num_points = semantic_labels_list[n].shape[0]
                with h5py.File(rawh5f_fn,'w') as h5f:
                    raw_h5f = Raw_H5f(h5f,rawh5f_fn,'SCANNET')
                    raw_h5f.set_num_default_row(num_points)
                    raw_h5f.append_to_dset('xyz',scene_points_list[n])
                    raw_h5f.append_to_dset('label_category',semantic_labels_list[n])
                    raw_h5f.create_done()

    def GenObj_RawH5f(self,k0=0,k1=1):
        for k in range(k0,k1):
            file_name = self.rawh5f_dir + '/%s_%d.rh5'%( self.split,k )
            xyz_cut_rate= [0,0,0.9]
            with h5py.File(file_name,'r') as h5f:
                rawh5f = Raw_H5f(h5f,file_name)
                rawh5f.generate_objfile(IsLabelColor=False,xyz_cut_rate=xyz_cut_rate)

    def SortRaw(self,block_step_xyz,MultiProcess=0):
        t0 = time.time()
        rawh5_file_ls = glob.glob( os.path.join( self.rawh5f_dir,'*.rh5' ) )
        rawh5_file_ls.sort()
        sorted_path = self.scans_h5f_dir + '/'+get_stride_step_name(block_step_xyz,block_step_xyz) + '/'+self.split
        IsShowInfoFinished = True

        IsMultiProcess = MultiProcess>1
        if not IsMultiProcess:
            WriteSortH5f_FromRawH5f(rawh5_file_ls,block_step_xyz,sorted_path,IsShowInfoFinished)
        else:
            pool = mp.Pool(MultiProcess)
            for rawh5f_fn in rawh5_file_ls:
                results = pool.apply_async(WriteSortH5f_FromRawH5f,([rawh5f_fn],block_step_xyz,sorted_path,IsShowInfoFinished))
            pool.close()
            pool.join()

            success_fns = []
            success_N = len(rawh5_file_ls)
            try:
                for k in range(success_N):
                    success_fns.append(results.get(timeout=0.1))
            except:
                assert len(success_fns)==success_N,"SortRaw failed. only %d files successed"%(len(success_fns))
            print("\n\nSortRaw:all %d files successed\n******************************\n"%(len(success_fns)))
        print('sort raw t= %f'%(time.time()-t0))

    def GenPyramid(self, base_stride, base_step, MultiProcess=0):
        file_list = []
        sh5f_dir = self.scans_h5f_dir+'/%s'%(get_stride_step_name(base_stride,base_step)) + '/' + self.split
        file_list += glob.glob( os.path.join( sh5f_dir, '*.sh5' ) )
        file_list.sort()

        IsMultiProcess = MultiProcess>1
        if IsMultiProcess:
            pool = mp.Pool(MultiProcess)
        for fn in file_list:
            if not IsMultiProcess:
                GenPyramidSortedFlie(fn)
            else:
                results = pool.apply_async(GenPyramidSortedFlie,(fn,))
        if IsMultiProcess:
            pool.close()
            pool.join()

            success_fns = []
            success_N = len(file_list)
            try:
                for k in range(success_N):
                    success_fns.append(results.get(timeout=0.1))
            except:
                assert len(success_fns)==success_N,"Norm failed. only %d files successed"%(len(success_fns))
            print("\n\n GenPyramid: all %d files successed\n******************************\n"%(len(success_fns)))

    def MergeNormed(self):
        plnh5_folder_name = 'stride_0d1_step_0d1_pl_nh5_1d6_2'
        #bxmh5_folder_name = 'stride_0d1_step_0d1_bmap_nh5_12800_1d6_2_fmn3-600_64_24-60_16_12-0d2_0d6_1d2-0d2_0d6_1d2'
        #bxmh5_folder_name = 'stride_0d1_step_0d1_bmap_nh5_12800_1d6_2_fmn6-2048_256_64-48_32_16-0d2_0d6_1d2-0d1_0d3_0d6'
        bxmh5_folder_name = 'stride_0d1_step_0d1_bmap_nh5_12800_1d6_2_fmn3-512_64_24-48_16_12-0d2_0d6_1d2-0d2_0d6_1d2'
        nh5_folder_names = [ plnh5_folder_name, bxmh5_folder_name]
        formats = ['.nh5','.bxmh5']
        pl_base_fn_ls = []
        if True:
            pl_region_h5f_path = self.scans_h5f_dir + '/' + nh5_folder_names[0] + '/' + self.split
            plfn_ls = glob.glob( pl_region_h5f_path + '/*' +  formats[0] )
            plfn_ls.sort()
            nonvoid_plfn_ls = []
            bxmh5_fn_ls = []
            for pl_fn in plfn_ls:
                is_intact, ck_str = Normed_H5f.check_nh5_intact( pl_fn )
                if not is_intact:
                    print(' ! ! ! Abort merging %s not intact: %s'%(house_name+formats[0], pl_fn))
                    continue
                if ck_str == 'void file':
                    if not SHOWONLYERR: print('void file: %s'%(pl_fn))
                    continue
                region_name = os.path.splitext(os.path.basename( pl_fn ))[0]
                bxmh5_fn = self.scans_h5f_dir + '/' + nh5_folder_names[1] + '/' + self.split + '/' + region_name + formats[1]
                if not os.path.exists( bxmh5_fn ):
                    print(' ! ! ! Abort merging %s not intact: %s'%(house_name+formats[0], pl_fn))
                    return
                with h5py.File( pl_fn, 'r' ) as plh5f, h5py.File( bxmh5_fn, 'r' ) as bxmh5f:
                    if not plh5f['data'].shape[0] == bxmh5f['bidxmaps_flatten'].shape[0]:
                        print('Abort merging %s \n  data shape (%d) != bidxmaps_flatten shape (%d): %s'%( pl_region_h5f_path, plh5f['data'].shape[0], bxmh5f['bidxmaps_flatten'].shape[0], pl_fn) )
                        return
                    else:
                        #print('shape match check ok: %s'%(region_name))
                        pass
                nonvoid_plfn_ls.append( pl_fn )
                bxmh5_fn_ls.append( bxmh5_fn )
            if len( nonvoid_plfn_ls )  == 0:
                print(  "no file, skip %s"%( house_name ) )
                return
            fn_ls = [ nonvoid_plfn_ls, bxmh5_fn_ls ]
            merged_file_names = ['','']

            for j in range(2):
                merged_path = os.path.dirname( self.scans_h5f_dir ) + '/each_house/' + nh5_folder_names[j] + '/'
                merged_file_names[j] = merged_path + self.split+formats[j]
                if not os.path.exists(merged_path):
                    os.makedirs(merged_path)
                MergeNormed_H5f( fn_ls[j], merged_file_names[j], IsShowSummaryFinished=True)
            # check after merged
            with h5py.File( merged_file_names[0], 'r' ) as plh5f, h5py.File( merged_file_names[1], 'r' ) as bxmh5f:
                if not plh5f['data'].shape[0] == bxmh5f['bidxmaps_flatten'].shape[0]:
                    print('! ! ! shape check failed:  data shape (%d) != bidxmaps_flatten shape (%d): \n\t%s \n\t%s'%( plh5f['data'].shape[0], bxmh5f['bidxmaps_flatten'].shape[0], merged_file_names[0],merged_file_names[1]) )
                else:
                    print( 'After merging, shape match check ok: %s'%(os.path.basename( merged_file_names[0] )) )
                    pass


def main(split):
        t0 = time.time()
        MultiProcess = 0
        scanet_prep = Scannet_Prepare(split)

        scanet_prep.Load_Raw_Scannet_Pickle()
        #scanet_prep.GenObj_RawH5f(0,3)
        base_step_stride = [0.1,0.1,0.1]
        scanet_prep.SortRaw( base_step_stride, MultiProcess )
        scanet_prep.GenPyramid(base_step_stride, base_step_stride, MultiProcess)
        #scanet_prep.MergeNormed()
        print('split = %s'%(split))
        print('T = %f sec'%(time.time()-t0))

if __name__ == '__main__':
    #main('test')
    main('train')
