# xyz
from __future__ import print_function
import pdb, traceback
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(BASE_DIR+'/scannet_meta')
from block_data_prep_util import Raw_H5f, Sort_RawH5f,Sorted_H5f,Normed_H5f,show_h5f_summary_info,MergeNormed_H5f,get_stride_step_name
from block_data_prep_util import GlobalSubBaseBLOCK,get_mean_sg_sample_rate,get_mean_flatten_sample_rate,check_h5fs_intact
import numpy as np
import h5py
import glob
import time
import multiprocessing as mp
import itertools
import pickle
from plyfile import PlyData, PlyElement
import json
import scannet_util

TMPDEBUG = True
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR,'data')
SCANNET_DATA_DIR = os.path.join(DATA_DIR, 'Scannet__H5F' )
SCANNET_MERGED_DATA_DIR = os.path.join(DATA_DIR, 'ScannetH5F' )

CLASS_NAMES = scannet_util.g_label_names
RAW2SCANNET = scannet_util.g_raw2scannet

def parse_scan_ply( ply_fn ):
    with open( ply_fn, 'r' ) as ply_fo:
        plydata = PlyData.read( ply_fo )
        num_ele = len(plydata.elements)
        num_vertex = plydata['vertex'].count
        num_face = plydata['face'].count
        data_vertex = plydata['vertex'].data
        data_face = plydata['face'].data

        ## face
        face_vertex_indices = data_face['vertex_indices']
        face_vertex_indices = np.concatenate(face_vertex_indices,axis=0)
        face_vertex_indices = np.reshape(face_vertex_indices,[-1,3])

        face_eles = ['vertex_indices']
        datas_face = {}
        for e in face_eles:
            datas_face[e] = np.expand_dims(data_face[e],axis=-1)

        ## vertex
        vertex_eles = ['x','y','z','red','green','blue','alpha']
        datas_vertex = {}
        for e in vertex_eles:
            datas_vertex[e] = np.expand_dims(data_vertex[e],axis=-1)
        vertex_xyz = np.concatenate([datas_vertex['x'],datas_vertex['y'],datas_vertex['z']],axis=1)
        vertex_rgb = np.concatenate([datas_vertex['red'],datas_vertex['green'],datas_vertex['blue']],axis=1)
        vertex_alpha = np.concatenate([datas_vertex['alpha']])
        points = np.concatenate( [vertex_xyz, vertex_rgb], -1 )

        return points

def parse_mesh_segs( mesh_segs_fn ):
    with open(mesh_segs_fn,'r') as jsondata:
        d = json.load(jsondata)
        mesh_seg = np.array( d['segIndices'] )
        #print len(mesh_seg)
    mesh_segid_to_pointid = {}
    for i in range(mesh_seg.shape[0]):
        if mesh_seg[i] not in mesh_segid_to_pointid:
            mesh_segid_to_pointid[mesh_seg[i]] = []
        mesh_segid_to_pointid[mesh_seg[i]].append(i)
    return mesh_segid_to_pointid, mesh_seg

def parse_aggregation( aggregation_fn ):
    with open( aggregation_fn,'r' ) as json_fo:
        d = json.load( json_fo )
        ids = []
        objectIds = []
        instance_segids = []
        labels = []
        for x in d['segGroups']:
            ids.append(x['id'])
            objectIds.append(x['objectId'])
            instance_segids.append(x['segments'])
            labels.append(x['label'])
        return instance_segids, labels

def parse_scan_raw( scene_name ):
    scene_name_base = os.path.basename( scene_name )
    ply_fn = scene_name + '/%s_vh_clean.ply'%(scene_name_base)
    mesh_segs_fn = scene_name + '/%s_vh_clean.segs.json'%(scene_name_base)
    aggregation_fn = scene_name + '/%s_vh_clean.aggregation.json'%(scene_name_base)

    segid_to_pointid, mesh_labels = parse_mesh_segs(mesh_segs_fn)
    points = parse_scan_ply( ply_fn )
    instance_segids, labels = parse_aggregation( aggregation_fn )

    # Each instance's points
    instance_points_list = []
    instance_labels_list = []
    semantic_labels_list = []
    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        instance_points = points[np.array(pointids),:]
        instance_points_list.append(instance_points)
        instance_labels_list.append(np.ones((instance_points.shape[0], 1))*i)
        if labels[i] not in RAW2SCANNET:
            label = 'unannotated'
        else:
            label = RAW2SCANNET[labels[i]]
        label = CLASS_NAMES.index(label)
        semantic_labels_list.append(np.ones((instance_points.shape[0], 1))*label)

    # Refactor data format
    scene_points = np.concatenate(instance_points_list, 0)
    instance_labels = np.concatenate(instance_labels_list, 0)
    semantic_labels = np.concatenate(semantic_labels_list, 0)

    return scene_points, instance_labels, semantic_labels, mesh_labels

def WriteRawH5f( scene_name, rawh5f_dir ):
    # save as rh5
    scene_name_base = os.path.basename( scene_name )
    rawh5f_fn = os.path.join(rawh5f_dir, scene_name_base+'.rh5')
    if Raw_H5f.check_rh5_intact( rawh5f_fn )[0]:
        print('rh5 intact: %s'%(rawh5f_fn))
        return scene_name
    print('start write rh5: %s'%(rawh5f_fn))

    scene_points, instance_labels, semantic_labels, mesh_labels = parse_scan_raw( scene_name )
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
    return scene_name

def WriteSortH5f_FromRawH5f(rawh5_file_ls,block_step_xyz,sorted_path,IsShowInfoFinished):
    Sort_RawH5f(rawh5_file_ls,block_step_xyz,sorted_path,IsShowInfoFinished)
    return rawh5_file_ls

def GenPyramidSortedFlie( fn ):
    with h5py.File(fn,'r') as f:
        sorted_h5f = Sorted_H5f(f,fn)
        Always_CreateNew_plh5 = False
        Always_CreateNew_bmh5 = False
        Always_CreateNew_bxmh5 = False
        if TMPDEBUG:
            Always_CreateNew_bmh5 = False
            Always_CreateNew_plh5 = False
            Always_CreateNew_bxmh5 = True

        sorted_h5f.file_saveas_pyramid_feed( IsShowSummaryFinished=True, Always_CreateNew_plh5 = Always_CreateNew_plh5, Always_CreateNew_bmh5 = Always_CreateNew_bmh5, Always_CreateNew_bxmh5=Always_CreateNew_bxmh5 )
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

def split_fn_ls_benchmark( plsph5_folder, bxmh5_folder, nonvoid_plfn_ls, bxmh5_fn_ls ):
    plsph5_folder = SCANNET_DATA_DIR + '/' + plsph5_folder
    bxmh5_folder = SCANNET_DATA_DIR + '/' + bxmh5_folder
    scannet_trainval_ls = list(np.loadtxt('./scannet_meta/scannet_trainval.txt','string'))
    scannet_test_ls = list(np.loadtxt('./scannet_meta/scannet_test.txt','string'))
    trainval_bxmh5_ls = [ os.path.join(bxmh5_folder, scene_name+'.bxmh5')  for scene_name in scannet_trainval_ls]
    trainval_sph5_ls = [ os.path.join(plsph5_folder, scene_name+'.sph5')  for scene_name in scannet_trainval_ls]
    test_bxmh5_ls = [ os.path.join(bxmh5_folder, scene_name+'.bxmh5')  for scene_name in scannet_test_ls]
    test_sph5_ls = [ os.path.join(plsph5_folder, scene_name+'.sph5')  for scene_name in scannet_test_ls]

    # check all file exist
    trainval_bxmh5_ls = [ fn for fn in trainval_bxmh5_ls if fn in bxmh5_fn_ls ]
    trainval_sph5_ls = [ fn for fn in trainval_sph5_ls if fn in nonvoid_plfn_ls ]
    test_bxmh5_ls = [ fn for fn in test_bxmh5_ls if fn in bxmh5_fn_ls ]
    test_sph5_ls = [ fn for fn in test_sph5_ls if fn in nonvoid_plfn_ls ]
    assert len(trainval_bxmh5_ls) ==  len(trainval_sph5_ls)
    assert len(test_bxmh5_ls) == len(test_sph5_ls)
    assert len(trainval_bxmh5_ls) ==  1201
    assert len(trainval_sph5_ls) == 1201
    assert len(test_bxmh5_ls) == 312
    assert len(test_sph5_ls) == 312
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
        scene_name_0 = os.path.splitext( os.path.basename(trainval_bxmh5_ls[k]) )[0]
        scene_name_1 = os.path.splitext( os.path.basename(trainval_bxmh5_ls[end-1]) )[0]
        scene_name_1 = scene_name_1[5:len(scene_name_1)]
        all_group_name_ls += ['trainval_'+scene_name_0+'_to_'+scene_name_1+'-'+str(end-k)]

    return [all_sph5_ls, all_bxmh5_ls], all_group_name_ls


class Scannet_Prepare():
    '''

    '''
    BasicDataDir = os.path.join( SCANNET_DATA_DIR,'BasicData' )

    def __init__(self):
        self.rawh5f_dir =  self.BasicDataDir+'/rawh5'

        self.sorted_path_stride_0d5_step_0d5 = os.path.join(SCANNET_DATA_DIR,'stride_0d5_step_0d5')
        self.sorted_path_stride_1_step_2 = os.path.join(SCANNET_DATA_DIR,'stride_1_step_2')
        self.sorted_path_stride_1_step_2_8192 = os.path.join(SCANNET_DATA_DIR,'stride_1_step_2')+'_8192'
        self.sorted_path_stride_1_step_2_8192_norm = os.path.join(SCANNET_DATA_DIR,'stride_1_step_2')+'_8192_normed'
        self.filename_stride_1_step_2_8192_norm_merged = os.path.join(SCANNET_DATA_DIR,'stride_1_step_2')+'_8192_normed.sph5'
        self.sorted_path_stride_2_step_4 = os.path.join(SCANNET_DATA_DIR,'stride_2_step_4')
        self.sorted_path_stride_2_step_4_8192 = os.path.join(SCANNET_DATA_DIR,'stride_2_step_4')+'_8192'
        self.sorted_path_stride_2_step_4_8192_norm = os.path.join(SCANNET_DATA_DIR,'stride_2_step_4')+'_8192_normed'
        self.filename_stride_2_step_4_8192_norm_merged = os.path.join(SCANNET_DATA_DIR,'stride_2_step_4')+'_8192_normed.sph5'

    def ParseRaw(self, MultiProcess):
        raw_path = DATA_DIR+'/scannet_data'

        rawh5f_dir = self.rawh5f_dir
        if not os.path.exists(rawh5f_dir):
            os.makedirs(rawh5f_dir)

        scene_name_ls =  glob.glob( raw_path+'/scene*' )
        scene_name_ls.sort()

        if MultiProcess < 2:
            for scene_name in scene_name_ls:
                WriteRawH5f( scene_name, rawh5f_dir )
        else:
            pool = mp.Pool(MultiProcess)
            for scene_name in scene_name_ls:
                results = pool.apply_async( WriteRawH5f, ( scene_name, rawh5f_dir))
            pool.close()
            pool.join()

            success_fns = []
            success_N = len(scene_name_ls)
            try:
                for k in range(success_N):
                    success_fns.append(results.get(timeout=0.1))
            except:
                assert len(success_fns)==success_N,"ParseRaw failed. only %d files successed"%(len(success_fns))
            print("\n\nParseRaw:all %d files successed\n******************************\n"%(len(success_fns)))

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
                # write one RawH5f file for one scane
                rawh5f_fn = os.path.join(rawh5f_dir,self.split+'_%d.rh5'%(n))
                if Raw_H5f.check_rh5_intact( rawh5f_fn )[0]:
                    print('rh5 intact: %s'%(rawh5f_fn))
                    continue
                num_points = semantic_labels_list[n].shape[0]
                with h5py.File(rawh5f_fn,'w') as h5f:
                    raw_h5f = Raw_H5f(h5f,rawh5f_fn,'SCANNET')
                    raw_h5f.set_num_default_row(num_points)
                    raw_h5f.append_to_dset('xyz',scene_points_list[n])
                    raw_h5f.append_to_dset('label_category',semantic_labels_list[n])
                    raw_h5f.create_done()


    def SortRaw(self,block_step_xyz,MultiProcess=0):
        t0 = time.time()
        rawh5_file_ls = glob.glob( os.path.join( self.rawh5f_dir,'*.rh5' ) )
        rawh5_file_ls.sort()
        sorted_path = self.BasicDataDir + '/'+get_stride_step_name(block_step_xyz,block_step_xyz)
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
        sh5f_dir = self.BasicDataDir+'/%s'%(get_stride_step_name(base_stride,base_step))
        file_list = glob.glob( os.path.join( sh5f_dir, '*.sh5' ) )
        file_list.sort()
        if TMPDEBUG:
            file_list = file_list[10:11]   # L
        #    #file_list = file_list[750:len(file_list)] # R
        #    #file_list = glob.glob( os.path.join( sh5f_dir, 'scene0062_01.sh5' ) )

        IsMultiProcess = MultiProcess>1
        if IsMultiProcess:
            pool = mp.Pool(MultiProcess)
        for k,fn in enumerate( file_list ):
            if not IsMultiProcess:
                GenPyramidSortedFlie(fn)
                print( 'Finish %d / %d files'%( k+1, len(file_list) ))
            else:
                results = pool.apply_async(GenPyramidSortedFlie,( fn,))
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
        plsph5_folder = 'ORG_sph5/90000_gs-4_-6d3'
        bxmh5_folder = 'ORG_bxmh5/90000_gs-4_-6d3_fmn6-6400_2400_320_32-32_16_32_48-0d1_0d3_0d9_2d7-0d1_0d2_0d6_1d8-pd3-4C0'

        sph5_folder_names = [ plsph5_folder, bxmh5_folder]
        formats = ['.sph5','.bxmh5']
        pl_base_fn_ls = []
        pl_region_h5f_path = SCANNET_DATA_DIR + '/' + sph5_folder_names[0]
        plfn_ls = glob.glob( pl_region_h5f_path + '/*' +  formats[0] )
        plfn_ls.sort()
        if len(plfn_ls) == 0:
            print('no file mathces %s'%(pl_region_h5f_path + '/*' +  formats[0] ))
        print('%d files found for %s'%(len(plfn_ls), pl_region_h5f_path + '/*' +  formats[0] ))

        nonvoid_plfn_ls = []
        bxmh5_fn_ls = []
        IsOnlyIntact = True
        for pl_fn in plfn_ls:
            is_intact, ck_str = Normed_H5f.check_sph5_intact( pl_fn )
            region_name = os.path.splitext(os.path.basename( pl_fn ))[0]
            if not is_intact:
                print(' ! ! ! Abort merging %s not intact: %s'%(formats[0], pl_fn))
                assert False
            if ck_str == 'void file':
                print('void file: %s'%(pl_fn))
                continue
            bxmh5_fn = SCANNET_DATA_DIR + '/' + sph5_folder_names[1] + '/' + region_name + formats[1]
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

        #allfn_ls, all_group_name_ls = split_fn_ls_benchmark( plsph5_folder, bxmh5_folder, nonvoid_plfn_ls, bxmh5_fn_ls )
        allfn_ls, all_group_name_ls = split_fn_ls( nonvoid_plfn_ls, bxmh5_fn_ls, merged_n=2 )

        for k in range( len(allfn_ls[0]) ):
            merged_file_names = ['','']

            for j in range(2):
                merged_path = SCANNET_MERGED_DATA_DIR + '/Merged' + sph5_folder_names[j][3:len(sph5_folder_names[j])] + '/'
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
    for fn in fn_ls:
        with h5py.File( fn,'r' ) as h5f:
            rawh5f = Raw_H5f(h5f,fn)
            rawh5f.generate_objfile(IsLabelColor=False,xyz_cut_rate=xyz_cut_rate)

def GenObj_sph5():
    #path = '/home/z/Research/dynamic_pointnet/data/Scannet__H5F/ORG_sph5/128000_gs-6_-10'
    path = '/home/z/Research/dynamic_pointnet/data/Scannet__H5F/ORG_sph5/60000_gs-3_-4d8'
    fn_ls = glob.glob( path+'/scene0000*.sph5' )
    for fn in fn_ls:
        with h5py.File(fn,'r') as h5f:
            normedh5f = Normed_H5f(h5f,fn)
            normedh5f.gen_gt_pred_obj_examples()

def main( ):
        t0 = time.time()
        MultiProcess = 0
        scanet_prep = Scannet_Prepare()

        #scanet_prep.ParseRaw( MultiProcess )
        base_step_stride = [0.1,0.1,0.1]
        #scanet_prep.SortRaw( base_step_stride, MultiProcess )
        scanet_prep.GenPyramid(base_step_stride, base_step_stride, MultiProcess)
        #scanet_prep.MergeNormed()
        print('T = %f sec'%(time.time()-t0))

if __name__ == '__main__':
    main()
    #GenObj_rh5()
    #GenObj_sph5()
