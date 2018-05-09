# xyz
from __future__ import print_function
import pdb, traceback
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR+'/all_datasets_meta')
from block_data_prep_util import Raw_H5f, Sort_RawH5f,Sorted_H5f,Normed_H5f,show_h5f_summary_info,MergeNormed_H5f,get_stride_step_name
from block_data_prep_util import GlobalSubBaseBLOCK,get_mean_sg_sample_rate,get_mean_flatten_sample_rate,check_h5fs_intact
import numpy as np
import h5py
import glob
import time
import multiprocessing as mp
import itertools
import pickle
import json
from  datasets_meta import DatasetsMeta
import geometric_util as geo_util

TMPDEBUG = False
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR,'data')

DATASETS = ['MATTERPORT', 'SCANNET', 'ETH']
for ds in DATASETS:
    sys.path.append('%s/%s_util'%(BASE_DIR,ds))

DATASET = 'SCANNET'
DATASET = 'ETH'
DS_Meta = DatasetsMeta( DATASET )

ORG_DATA_DIR = os.path.join(DATA_DIR, DATASET+'__H5F' )
MERGED_DATA_DIR = os.path.join(DATA_DIR, DATASET+'H5F' )

#CLASS_NAMES = DS_Meta.label_names

def WriteSortH5f_FromRawH5f(rawh5_file_ls,block_step_xyz,sorted_path, RotateBeforeSort, IsShowInfoFinished):
    Sort_RawH5f(rawh5_file_ls,block_step_xyz,sorted_path,RotateBeforeSort, IsShowInfoFinished)
    return rawh5_file_ls

def GenPyramidSortedFlie( fn, data_aug_configs ):

    with h5py.File(fn,'r') as f:
        sorted_h5f = Sorted_H5f(f,fn)
        Always_CreateNew_plh5 = False
        Always_CreateNew_bmh5 = False
        Always_CreateNew_bxmh5 = False
        if TMPDEBUG:
            Always_CreateNew_bmh5 = False
            Always_CreateNew_plh5 = False
            Always_CreateNew_bxmh5 = False

        sorted_h5f.file_saveas_pyramid_feed(
                            IsShowSummaryFinished=True,
                            Always_CreateNew_plh5 = Always_CreateNew_plh5,
                            Always_CreateNew_bmh5 = Always_CreateNew_bmh5,
                            Always_CreateNew_bxmh5=Always_CreateNew_bxmh5,
                            IsGenPly=False,
                            data_aug_configs = data_aug_configs )
    return fn

def split_fn_ls( nonvoid_plfn_ls, bxmh5_fn_ls, merged_n=2 ):
    nf = len(nonvoid_plfn_ls)
    merged_n = min( merged_n, nf )
    group_n = int( nf/merged_n )
    allfn_ls = [ [], [] ]
    all_group_name_ls = []
    for i in range( 0, nf, group_n ):
        end = min( nf, i+group_n )
        allfn_ls[0].append( nonvoid_plfn_ls[i:end] )
        allfn_ls[1].append( bxmh5_fn_ls[i:end] )
        all_group_name_ls.append( '%d_%d'%(i, end) )
    return allfn_ls, all_group_name_ls

def split_fn_ls_benchmark( plsph5_folder, bxmh5_folder, nonvoid_plfn_ls, bxmh5_fn_ls, void_f_n ):
    plsph5_folder = ORG_DATA_DIR + '/' + plsph5_folder
    bxmh5_folder = ORG_DATA_DIR + '/' + bxmh5_folder
    scannet_trainval_ls = list(np.loadtxt('./SCANNET_util/scannet_trainval.txt','string'))
    scannet_test_ls = list(np.loadtxt('./SCANNET_util/scannet_test.txt','string'))
    trainval_bxmh5_ls = [ os.path.join(bxmh5_folder, fn+'.bxmh5')  for fn in scannet_trainval_ls]
    trainval_sph5_ls = [ os.path.join(plsph5_folder, fn+'.sph5')  for fn in scannet_trainval_ls]
    test_bxmh5_ls = [ os.path.join(bxmh5_folder, fn+'.bxmh5')  for fn in scannet_test_ls]
    test_sph5_ls = [ os.path.join(plsph5_folder, fn+'.sph5')  for fn in scannet_test_ls]

    # check all file exist
    trainval_bxmh5_ls = [ fn for fn in trainval_bxmh5_ls if fn in bxmh5_fn_ls ]
    trainval_sph5_ls = [ fn for fn in trainval_sph5_ls if fn in nonvoid_plfn_ls ]
    test_bxmh5_ls = [ fn for fn in test_bxmh5_ls if fn in bxmh5_fn_ls ]
    test_sph5_ls = [ fn for fn in test_sph5_ls if fn in nonvoid_plfn_ls ]
    assert len(trainval_bxmh5_ls) ==  len(trainval_sph5_ls)
    assert len(test_bxmh5_ls) == len(test_sph5_ls)
    if void_f_n==0:
        assert len(trainval_bxmh5_ls) ==  1201
        assert len(trainval_sph5_ls) == 1201
        assert len(test_bxmh5_ls) == 312
        assert len(test_sph5_ls) == 312
    assert len(trainval_bxmh5_ls) + len(test_bxmh5_ls) + void_f_n == 1201 + 312
    trainval_bxmh5_ls.sort()
    trainval_sph5_ls.sort()
    test_bxmh5_ls.sort()
    test_sph5_ls.sort()

    all_bxmh5_ls = [test_bxmh5_ls]
    all_sph5_ls = [test_sph5_ls]
    all_group_name_ls = ['test']
    # split trainval ls
    group_n = 301
    for k in range( 0, len(trainval_bxmh5_ls), group_n ):
        end  = min( k+group_n, len(trainval_bxmh5_ls) )
        all_bxmh5_ls += [trainval_bxmh5_ls[k:end]]
        all_sph5_ls += [trainval_sph5_ls[k:end]]
        fn_0 = os.path.splitext( os.path.basename(trainval_bxmh5_ls[k]) )[0]
        fn_1 = os.path.splitext( os.path.basename(trainval_bxmh5_ls[end-1]) )[0]
        fn_1 = fn_1[5:len(fn_1)]
        all_group_name_ls += ['trainval_'+fn_0+'_to_'+fn_1+'-'+str(end-k)]

    return [all_sph5_ls, all_bxmh5_ls], all_group_name_ls


def WriteRawH5f( fn, rawh5f_dir ):
    if DATASET == 'SCANNET':
        return WriteRawH5f_SCANNET( fn, rawh5f_dir )
    elif DATASET == 'ETH':
        return WriteRawH5f_ETH( fn, rawh5f_dir )

def WriteRawH5f_SCANNET( fn, rawh5f_dir ):
    # save as rh5
    import SCANNET_util
    fn_base = os.path.basename( fn )
    rawh5f_fn = os.path.join(rawh5f_dir, fn_base+'.rh5')
    if Raw_H5f.check_rh5_intact( rawh5f_fn )[0]:
        print('rh5 intact: %s'%(rawh5f_fn))
        return fn
    print('start write rh5: %s'%(rawh5f_fn))

    scene_points, instance_labels, semantic_labels, mesh_labels = SCANNET_util.parse_raw_SCANNET( fn )
    num_points = scene_points.shape[0]
    with h5py.File(rawh5f_fn,'w') as h5f:
        raw_h5f = Raw_H5f(h5f,rawh5f_fn,'SCANNET')
        raw_h5f.set_num_default_row(num_points)
        raw_h5f.append_to_dset('xyz', scene_points[:,0:3])
        raw_h5f.append_to_dset('color', scene_points[:,3:6])
        raw_h5f.append_to_dset('label_category', semantic_labels)
        raw_h5f.append_to_dset('label_instance', instance_labels)
        raw_h5f.append_to_dset('label_mesh', mesh_labels)
        raw_h5f.rh5_create_done()
    return fn

def WriteRawH5f_ETH( fn_7z, rawh5f_dir ):
    import ETH_util
    fn_base = os.path.basename( fn_7z )
    fn_base = os.path.splitext( fn_base )[0]
    if fn_base[-3:] == 'txt':
        fn_base = os.path.splitext( fn_base )[0]
    rawh5f_fn = os.path.join(rawh5f_dir, fn_base+'.rh5')
    if Raw_H5f.check_rh5_intact( rawh5f_fn )[0]:
        print('rh5 intact: %s'%(rawh5f_fn))
        return fn_7z
    print('start write rh5: %s'%(rawh5f_fn))

    xyz, intensity, rgb, labels = ETH_util.parse_raw_ETH( fn_7z )
    num_points = xyz.shape[0]

    with h5py.File( rawh5f_fn, 'w' ) as h5f:
        raw_h5f = Raw_H5f(h5f,rawh5f_fn,'ETH')
        raw_h5f.set_num_default_row(num_points)
        raw_h5f.append_to_dset('xyz', xyz)
        raw_h5f.append_to_dset('color', rgb)
        raw_h5f.append_to_dset('intensity', intensity)
        if labels != None:
            raw_h5f.append_to_dset( 'label_category', labels )
        raw_h5f.rh5_create_done()
    print('finish : %s'%(rawh5f_fn))
    return rawh5f_fn


class H5Prepare():
    '''

    '''
    BasicDataDir = os.path.join( ORG_DATA_DIR,'BasicData' )

    def __init__(self):
        self.rawh5f_dir =  self.BasicDataDir+'/rawh5'

    def ParseRaw(self, MultiProcess):
        raw_path = './' + DATASET

        rawh5f_dir = self.rawh5f_dir
        if not os.path.exists(rawh5f_dir):
            os.makedirs(rawh5f_dir)

        if DATASET == 'SCANNET':
            glob_fn = raw_path+'/scene*'
        elif DATASET == 'ETH':
            glob_fn = raw_path+'/*.7z'

        fn_ls =  glob.glob( glob_fn )
        fn_ls.sort()
        if len(fn_ls) == 0:
            print('no file matches %s'%( glob_fn ))

        if TMPDEBUG:
            fn_ls = fn_ls[0:1]

        if MultiProcess < 2:
            for fn in fn_ls:
                WriteRawH5f( fn, rawh5f_dir )
        else:
            pool = mp.Pool(MultiProcess)
            for fn in fn_ls:
                results = pool.apply_async( WriteRawH5f, ( fn, rawh5f_dir))
            pool.close()
            pool.join()

            success_fns = []
            success_N = len(fn_ls)
            try:
                for k in range(success_N):
                    success_fns.append(results.get(timeout=0.1))
            except:
                assert len(success_fns)==success_N,"ParseRaw failed. only %d files successed"%(len(success_fns))
            print("\n\nParseRaw:all %d files successed\n******************************\n"%(len(success_fns)))



    def SortRaw(self, block_step_xyz, MultiProcess=0 , RxyzBeforeSort=None ):
        t0 = time.time()
        rawh5_file_ls = glob.glob( os.path.join( self.rawh5f_dir,'*.rh5' ) )
        rawh5_file_ls.sort()
        sorted_path = self.BasicDataDir + '/'+get_stride_step_name(block_step_xyz,block_step_xyz)
        if type(RxyzBeforeSort)!=type(None) and np.sum(RxyzBeforeSort==0)!=0:
            RotateBeforeSort = geo_util.EulerRotate( RxyzBeforeSort, 'xyz' )
            rdgr = RxyzBeforeSort * 180/np.pi
            RotateBeforeSort_str = '-R_%d_%d_%d'%( rdgr[0], rdgr[1], rdgr[2] )
            sorted_path += RotateBeforeSort_str
        else:
            RotateBeforeSort = None
        IsShowInfoFinished = True

        IsMultiProcess = MultiProcess>1
        if not IsMultiProcess:
            WriteSortH5f_FromRawH5f(rawh5_file_ls,block_step_xyz,sorted_path, RotateBeforeSort, IsShowInfoFinished)
        else:
            pool = mp.Pool(MultiProcess)
            for rawh5f_fn in rawh5_file_ls:
                results = pool.apply_async(WriteSortH5f_FromRawH5f,([rawh5f_fn],block_step_xyz,sorted_path, RotateBeforeSort, IsShowInfoFinished))
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

    def GenPyramid(self, base_stride, base_step, data_aug_configs, MultiProcess=0):
        sh5f_dir = self.BasicDataDir+'/%s'%(get_stride_step_name(base_stride,base_step))
        file_list = glob.glob( os.path.join( sh5f_dir, '*.sh5' ) )
        file_list.sort()
        if TMPDEBUG:
            choice = range(0,800,10)[0:min(8,len(file_list))]
            file_list = [ file_list[c] for c in choice ]
            #file_list = file_list[0:750]   # L
            #file_list = file_list[750:len(file_list)] # R
            #file_list = glob.glob( os.path.join( sh5f_dir, 'scene0509_00.sh5' ) )

        IsMultiProcess = MultiProcess>1
        if IsMultiProcess:
            pool = mp.Pool(MultiProcess)
        for k,fn in enumerate( file_list ):
            if not IsMultiProcess:
                GenPyramidSortedFlie(fn, data_aug_configs)
                print( 'Finish %d / %d files'%( k+1, len(file_list) ))
            else:
                results = pool.apply_async(GenPyramidSortedFlie,( fn,data_aug_configs))
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



    def MergeNormed(self, data_aug_configs):
        plsph5_folder = 'ORG_sph5/30000_gs-2d4_-3d4-du'
        bxmh5_folder = 'ORG_bxmh5/30000_gs-2d4_-3d4_fmn1444-2048_1024_128_24-48_32_48_27-0d1_0d4_1_2d2-0d1_0d2_0d6_1d2-pd3-mbf-4B1-du'

        sph5_folder_names = [ plsph5_folder, bxmh5_folder]
        formats = ['.sph5','.bxmh5']
        pl_base_fn_ls = []
        pl_region_h5f_path = ORG_DATA_DIR + '/' + sph5_folder_names[0]
        plfn_ls = glob.glob( pl_region_h5f_path + '/*' +  formats[0] )
        plfn_ls.sort()
        if len(plfn_ls) == 0:
            print('no file mathces %s'%(pl_region_h5f_path + '/*' +  formats[0] ))
        print('%d files found for %s'%(len(plfn_ls), pl_region_h5f_path + '/*' +  formats[0] ))

        nonvoid_plfn_ls = []
        bxmh5_fn_ls = []
        void_f_n = 0
        IsOnlyIntact = True
        for pl_fn in plfn_ls:
            is_intact, ck_str = Normed_H5f.check_sph5_intact( pl_fn )
            region_name = os.path.splitext(os.path.basename( pl_fn ))[0]
            if not is_intact:
                print(' ! ! ! Abort merging %s not intact: %s'%(formats[0], pl_fn))
                assert False
            if ck_str == 'void file':
                print('void file: %s'%(pl_fn))
                void_f_n += 1
                continue
            bxmh5_fn = ORG_DATA_DIR + '/' + sph5_folder_names[1] + '/' + region_name + formats[1]
            if not os.path.exists( bxmh5_fn ):
                if IsOnlyIntact: continue
                print(' ! ! ! Abort merging %s not exist: %s'%(formats[0], bxmh5_fn))
                assert False
            with h5py.File( pl_fn, 'r' ) as plh5f, h5py.File( bxmh5_fn, 'r' ) as bxmh5f:
                if not plh5f['data'].shape[0] == bxmh5f['bidxmaps_flat'].shape[0]:
                    print('Abort merging %s \n  data shape (%d) != bidxmaps_flat shape (%d): %s'%( pl_region_h5f_path, plh5f['data'].shape[0], bxmh5f['bidxmaps_flat'].shape[0], pl_fn) )
                    assert False
                else:
                    #print('shape match check ok: %s'%(region_name))
                    pass
            nonvoid_plfn_ls.append( pl_fn )
            bxmh5_fn_ls.append( bxmh5_fn )
        if len( nonvoid_plfn_ls )  == 0:
            print(  "no file, skip merging" )
            return

        #allfn_ls, all_group_name_ls = split_fn_ls_benchmark( plsph5_folder, bxmh5_folder, nonvoid_plfn_ls, bxmh5_fn_ls, void_f_n )
        allfn_ls, all_group_name_ls = split_fn_ls( nonvoid_plfn_ls, bxmh5_fn_ls, merged_n=2 )

        for k in range( len(allfn_ls[0]) ):
            merged_file_names = ['','']

            for j in range(2):
                merged_path = MERGED_DATA_DIR + '/Merged' + sph5_folder_names[j][3:len(sph5_folder_names[j])] + '/'
                merged_file_names[j] = merged_path + all_group_name_ls[k] + formats[j]
                if not os.path.exists(merged_path):
                    os.makedirs(merged_path)
                MergeNormed_H5f( allfn_ls[j][k], merged_file_names[j], IsShowSummaryFinished=True)
            # check after merged
            with h5py.File( merged_file_names[0], 'r' ) as plh5f, h5py.File( merged_file_names[1], 'r' ) as bxmh5f:
                if not plh5f['data'].shape[0] == bxmh5f['bidxmaps_flat'].shape[0]:
                    print('! ! ! shape check failed:  data shape (%d) != bidxmaps_flat shape (%d): \n\t%s \n\t%s'%( plh5f['data'].shape[0], bxmh5f['bidxmaps_flat'].shape[0], merged_file_names[0],merged_file_names[1]) )
                else:
                    print( 'After merging, shape match check ok: %s'%(os.path.basename( merged_file_names[0] )) )
                    pass

def GenObj_rh5():
    xyz_cut_rate= [0,0,0.9]
    xyz_cut_rate= None

    path = '/home/z/Research/dynamic_pointnet/data/Scannet__H5F/BasicData/rawh5'
    fn_ls = glob.glob( path+'/scene0002*.rh5' )

    path = '/home/z/Research/dynamic_pointnet/data/ETH__H5F/BasicData/rawh5'
    fn_ls = glob.glob( path+'/marketplacefeldkirch_station7_intensity_rgb.rh5' )
    fn_ls = glob.glob( path+'/StGallenCathedral_station6_rgb_intensity-reduced.rh5' )

    for fn in fn_ls:
        if not Raw_H5f.check_rh5_intact( fn )[0]:
            print('rh5 not intact, abort gen obj')
            return
        with h5py.File( fn,'r' ) as h5f:
            rawh5f = Raw_H5f(h5f,fn)
            rawh5f.generate_objfile(IsLabelColor=False,xyz_cut_rate=xyz_cut_rate)

def GenObj_sh5():
    path = '/home/z/Research/dynamic_pointnet/data/Scannet__H5F/BasicData/stride_0d1_step_0d1'
    fn_ls = glob.glob( path+'/scene0000_00.sh5' )
    for fn in fn_ls:
        with h5py.File( fn,'r' ) as h5f:
            sh5f = Sorted_H5f(h5f,fn)
            sh5f.gen_file_obj(IsLabelColor=False)


def GenObj_sph5():
    path = '/home/z/Research/dynamic_pointnet/data/Scannet__H5F/ORG_sph5/30000_gs-2d4_-3d4'
    path = '/home/z/Research/dynamic_pointnet/data/Scannet__H5F/ORG_sph5/30000_gs-2d4_-3d4-dec5'
    fn_ls = glob.glob( path+'/scene*.sph5' )
    for fn in fn_ls:
        with h5py.File(fn,'r') as h5f:
            normedh5f = Normed_H5f(h5f,fn)
            normedh5f.gen_gt_pred_obj_examples()

def main( ):
        t0 = time.time()
        MultiProcess = 6
        h5prep = H5Prepare()

        h5prep.ParseRaw( MultiProcess )
        base_step_stride = [0.1,0.1,0.1]
        #RxyzBeforeSort = np.array([0,0,45])*np.pi/180
        RxyzBeforeSort = None
        #h5prep.SortRaw( base_step_stride, MultiProcess, RxyzBeforeSort )

        data_aug_configs = {}
        data_aug_configs['delete_unlabelled'] = True
        #data_aug_configs['delete_easy_categories_num'] = 5

        #h5prep.GenPyramid(base_step_stride, base_step_stride, data_aug_configs,  MultiProcess)
        #h5prep.MergeNormed( data_aug_configs )
        print('T = %f sec'%(time.time()-t0))

if __name__ == '__main__':
    main()
    #GenObj_rh5()
    #GenObj_sph5()
    #GenObj_sh5()
