#xyz Dec 2017
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
import zipfile,gzip
from plyfile import PlyData, PlyElement

TMPDEBUG = True

ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR,'data')
DATA_SOURCE= 'scannet_data'
SCANNET_DATA_DIR = os.path.join(DATA_DIR,DATA_SOURCE)

def zip_extract(ply_item_name,zipf,house_dir_extracted):
    '''
    extract file if not already
    '''
    #zipfile_name = '%s/%s/%s.%s'%(house_name,groupe_name,file_name,file_format)
    file_path = house_dir_extracted + '/' + ply_item_name
    if not os.path.exists(file_path):
        print('extracting %s...'%(file_path))
        file_path_extracted  = zipf.extract(ply_item_name,house_dir_extracted)
        print('file extracting finished: %s'%(file_path_extracted) )
        assert file_path == file_path_extracted
    else:
        print('file file already extracted: %s'%(file_path))
    return file_path

def parse_ply_file(ply_fo,IsDelVexMultiSem):
    '''
    element vertex 1522546
    property float x
    property float y
    property float z
    property float nx
    property float ny
    property float nz
    property float tx
    property float ty
    property uchar red
    property uchar green
    property uchar blue

    element face 3016249
    property list uchar int vertex_indices
    property int material_id
    property int segment_id
    property int category_id
    '''
    plydata = PlyData.read(ply_fo)
    num_ele = len(plydata.elements)
    num_vertex = plydata['vertex'].count
    num_face = plydata['face'].count
    data_vertex = plydata['vertex'].data
    data_face = plydata['face'].data

    ## face
    face_vertex_indices = data_face['vertex_indices']
    face_vertex_indices = np.concatenate(face_vertex_indices,axis=0)
    face_vertex_indices = np.reshape(face_vertex_indices,[-1,3])

    face_eles = ['vertex_indices','material_id','segment_id','category_id']
    datas_face = {}
    for e in face_eles:
        datas_face[e] = np.expand_dims(data_face[e],axis=-1)
    face_semantic = np.concatenate([datas_face['category_id'],datas_face['segment_id'],datas_face['material_id']],axis=1)

    ## vertex
    vertex_eles = ['x','y','z','nx','ny','nz','tx','ty','red','green','blue']
    datas_vertex = {}
    for e in vertex_eles:
        datas_vertex[e] = np.expand_dims(data_vertex[e],axis=-1)
    vertex_xyz = np.concatenate([datas_vertex['x'],datas_vertex['y'],datas_vertex['z']],axis=1)
    vertex_nxnynz = np.concatenate([datas_vertex['nx'],datas_vertex['ny'],datas_vertex['nz']],axis=1)
    vertex_rgb = np.concatenate([datas_vertex['red'],datas_vertex['green'],datas_vertex['blue']],axis=1)

    vertex_semantic,vertex_indices_multi_semantic,face_indices_multi_semantic = get_vertex_label_from_face(face_vertex_indices,face_semantic,num_vertex)

    if IsDelVexMultiSem:
        vertex_xyz = np.delete(vertex_xyz,vertex_indices_multi_semantic,axis=0)
        vertex_nxnynz = np.delete(vertex_nxnynz,vertex_indices_multi_semantic,axis=0)
        vertex_rgb = np.delete(vertex_rgb,vertex_indices_multi_semantic,axis=0)
        vertex_semantic = np.delete(vertex_semantic,vertex_indices_multi_semantic,axis=0)
        face_vertex_indices = np.delete(face_vertex_indices,face_indices_multi_semantic,axis=0)
        face_semantic = np.delete(face_semantic,face_indices_multi_semantic,axis=0)

    return vertex_xyz,vertex_nxnynz,vertex_rgb,vertex_semantic,face_vertex_indices,face_semantic

def parse_house_file(house_fo):
    for i,line in enumerate( house_fo ):
        if i<1:
            print(line)
        break
def get_vertex_label_from_face(face_vertex_indices,face_semantic,num_vertex):
    '''
    face_vertex_indices: the vertex indices in each face
    vertex_face_indices: the face indices in each vertex
    '''
    vertex_face_indices = -np.ones(shape=[num_vertex,30])
    face_num_per_vertex = np.zeros(shape=[num_vertex]).astype(np.int8)
    vertex_semantic = np.zeros(shape=[num_vertex,3]) # only record the first one
    vertex_semantic_num = np.zeros(shape=[num_vertex])
    vertex_indices_multi_semantic = set()
    face_indices_multi_semantic = set()
    for i in range(face_vertex_indices.shape[0]):
        for vertex_index in face_vertex_indices[i]:
            face_num_per_vertex[vertex_index] += 1
            vertex_face_indices[vertex_index,face_num_per_vertex[vertex_index]-1] = i

            if vertex_semantic_num[vertex_index] == 0:
                vertex_semantic_num[vertex_index] += 1
                vertex_semantic[vertex_index] = face_semantic[i]
            else:
                # (1) Only 60% vertexs have unique labels for all three semntics
                # (2) There are 96% vertexs have unique labels for the first two:  category_id and segment_id
                IsSameSemantic = (vertex_semantic[vertex_index][0:2]==face_semantic[i][0:2]).all()
                if not IsSameSemantic:
                    vertex_semantic_num[vertex_index] += 1
                    vertex_indices_multi_semantic.add(vertex_index)
                    face_indices_multi_semantic.add(i)
    vertex_indices_multi_semantic = np.array(list(vertex_indices_multi_semantic))
    face_indices_multi_semantic = np.array(list(face_indices_multi_semantic))
    print('vertex rate with multiple semantic: %f'%(1.0*vertex_indices_multi_semantic.shape[0]/num_vertex))

   # vertex_semantic_num_max = np.max(vertex_semantic_num)
   # vertex_semantic_num_min = np.min(vertex_semantic_num)
   # vertex_semantic_num_mean = np.mean(vertex_semantic_num)
   # vertex_semantic_num_one = np.sum(vertex_semantic_num==1)
   # print(vertex_semantic_num_max)
   # print(vertex_semantic_num_mean)
   # print(vertex_semantic_num_min)
   # print(1.0*vertex_semantic_num_one/num_vertex)

    return vertex_semantic,vertex_indices_multi_semantic,face_indices_multi_semantic

def WriteRawH5f_Region_Ply(ply_item_name,rs_zf,house_name,house_h5f_dir,house_dir_extracted):
    #file_name = 'region'+str(k_region)
    region_ply_fn = zip_extract(ply_item_name,rs_zf,house_dir_extracted)
    s = ply_item_name.index('region_segmentations/region')+len('region_segmentations/region')
    e = ply_item_name.index('.ply')
    k_region = int( ply_item_name[ s:e ] )
    rawh5f_fn = house_h5f_dir+'/rawh5f/region'+str(k_region)+'.rh5'
    IsDelVexMultiSem = True
    IsIntact,_  = check_h5fs_intact(rawh5f_fn)
    if  IsIntact:
        print('file intact: %s'%(region_ply_fn))
    else:
        with open(region_ply_fn,'r') as ply_fo, h5py.File(rawh5f_fn,'w') as h5f:
            vertex_xyz,vertex_nxnynz,vertex_rgb,vertex_semantic,face_vertex_indices,face_semantic = parse_ply_file(ply_fo,IsDelVexMultiSem)

            raw_h5f = Raw_H5f(h5f,rawh5f_fn,'MATTERPORT')
            raw_h5f.set_num_default_row(vertex_xyz.shape[0])
            raw_h5f.append_to_dset('xyz',vertex_xyz)
            raw_h5f.append_to_dset('nxnynz',vertex_nxnynz)
            raw_h5f.append_to_dset('color',vertex_rgb)
            raw_h5f.append_to_dset('label_category',vertex_semantic[:,0]) # category_id
            raw_h5f.append_to_dset('label_instance',vertex_semantic[:,1]) # segment_id
            raw_h5f.append_to_dset('label_material',vertex_semantic[:,0]) # material_id
            raw_h5f.create_done()
            raw_h5f.show_h5f_summary_info()

    return region_ply_fn

def WriteSortH5f_FromRawH5f(rawh5_file_ls,block_step_xyz,sorted_path,IsShowInfoFinished):
    Sort_RawH5f(rawh5_file_ls,block_step_xyz,sorted_path,IsShowInfoFinished)
    return rawh5_file_ls

def MergeSampleNorm_FromSortedH5f( base_sorted_h5fname,new_stride,new_step,new_sorted_path,more_actions_config ):
    with h5py.File(base_sorted_h5fname,'r') as f:
        sorted_h5f = Sorted_H5f(f,base_sorted_h5fname)
        sorted_h5f.merge_to_new_step(new_stride,new_step,new_sorted_path,more_actions_config)
    return base_sorted_h5fname
def SampleFile(base_sorted_h5fname,numpoint_block,IsGenNorm):
    with h5py.File(base_sorted_h5fname,'r') as f:
        sorted_h5f = Sorted_H5f(f,base_sorted_h5fname)
        sorted_h5f.file_random_sampling(numpoint_block,gen_norm=IsGenNorm)
    return base_sorted_h5fname

def NormSortedSampledFlie(fn):
    with h5py.File(fn,'r') as f:
        sorted_h5f = Sorted_H5f(f,fn)
        sorted_h5f.file_normalization(True)
    return fn

def GenPyramidSortedFlie(fn):
    with h5py.File(fn,'r') as f:
        sorted_h5f = Sorted_H5f(f,fn)
        Always_CreateNew_pyh5 = False
        Always_CreateNew_bmh5 = False

        print(fn)
        sorted_h5f.file_saveas_pyramid_feed(True,Always_CreateNew_pyh5 = Always_CreateNew_pyh5, Always_CreateNew_bmh5 = Always_CreateNew_bmh5 )
    return fn

class Matterport3D_Prepare():
    '''
    Read each region as a h5f.
    Vertex and face are stored in seperate h5f.
    In vertex h5f, the corresponding face ids are listed as "face indices".
    In face h5f, the corresponding vertex ids are also listed as "vertex indices"
    While downsampling of vertex set, all the faces related with the deleted point are deleted. Thus, the downsampling rate should be very low.
        If want high downsamplig rate, it should start with possion reconstuction mesh of low depth.
    The semantic labels of each point are achieved from faces.
    '''

    matterport3D_root_dir = '/DS/Matterport3D/Matterport3D_Whole'
    matterport3D_extracted_dir = '/DS/Matterport3D/Matterport3D_Whole_extracted'
    matterport3D_h5f_dir = '/DS/Matterport3D/Matterport3D_H5F'

    def __init__(self,house_name = '17DRP5sb8fy',scans_name = '/v1/scans'):
        self.house_name = house_name
        self.scans_name = scans_name

        self.scans_dir = self.matterport3D_root_dir+scans_name
        self.house_dir = self.scans_dir+'/%s'%(house_name)
        self.region_segmentations_zip_fn = self.house_dir+'/region_segmentations.zip'

        self.scans_h5f_dir = self.matterport3D_h5f_dir+scans_name
        self.house_h5f_dir = self.scans_h5f_dir+'/%s'%(house_name)
        self.house_rawh5f_dir = self.house_h5f_dir+'/rawh5f'
        if not os.path.exists(self.house_rawh5f_dir):
            os.makedirs(self.house_rawh5f_dir)
        self.house_dir_extracted = self.matterport3D_extracted_dir+scans_name+'/%s'%(house_name)
        if not os.path.exists(self.house_dir_extracted):
            os.makedirs(self.house_dir_extracted)

    def Parse_house_regions(self,MultiProcess=0):
        t0 = time.time()
        rs_zf = zipfile.ZipFile(self.region_segmentations_zip_fn,'r')

        namelist_ply = [ name for name in rs_zf.namelist()  if 'ply' in name]
        num_region = len(namelist_ply)

        IsMultiProcess = MultiProcess>1
        if IsMultiProcess:
            pool = mp.Pool(MultiProcess)
        for ply_item_name in namelist_ply:
            if not IsMultiProcess:
                WriteRawH5f_Region_Ply(ply_item_name,rs_zf,self.house_name,self.house_h5f_dir,self.house_dir_extracted)
            else:
                results = pool.apply_async(WriteRawH5f_Region_Ply,(ply_item_name,rs_zf,self.house_name,self.house_h5f_dir,self.house_dir_extracted))
                s = ply_item_name.index('region_segmentations/region')+len('region_segmentations/region')
                e = ply_item_name.index('.ply')
                k_region = int( ply_item_name[ s:e ] )
                print('apply_async %d'%(k_region))
        if IsMultiProcess:
            pool.close()
            pool.join()

            success_fns = []
            success_N = num_region
            try:
                for k in range(success_N):
                    success_fns.append(results.get(timeout=0.1))
            except:
                assert len(success_fns)==success_N,"Parse_house_regions failed. only %d files successed"%(len(success_fns))
            print("\n\n Parse_house_regions:all %d files successed\n******************************\n"%(len(success_fns)))

        print('Parse house time: %f'%(time.time()-t0))

    def SortRaw(self,block_step_xyz,MultiProcess=0):
        t0 = time.time()
        rawh5_file_ls = glob.glob(self.house_h5f_dir+'/rawh5f/*.rh5')
        #block_step_xyz = [0.5,0.5,0.5]
        sorted_path = self.house_h5f_dir+'/'+get_stride_step_name(block_step_xyz,block_step_xyz)
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

    def MergeSampleNorm(self,base_step_stride,new_stride,new_step,numpoint_block,MultiProcess=0):
        '''
         1 merge to new block step/stride size
             obj_merged: generate obj for merged
         2 randomly sampling to fix point number in each block
             obj_sampled_merged
         3 normalizing sampled block
        '''
        base_stride = base_step = base_step_stride
        base_path = self.house_h5f_dir+'/'+get_stride_step_name(base_stride,base_step)
        new_stride = new_stride
        new_step = new_step
        new_sorted_path = self.house_h5f_dir+'/'+get_stride_step_name(new_stride,new_step)


        base_file_list = glob.glob( os.path.join(base_path,'*.sh5') )
        print('%d sh5 files in %s'%(len(base_file_list),base_path))

        more_actions_config = {}
        more_actions_config['actions'] = []
        more_actions_config['actions'] = ['sample_merged']
        #more_actions_config['actions'] = ['sample_merged','norm_sampled_merged']
        more_actions_config['sample_num'] = numpoint_block

        IsMultiProcess = MultiProcess>1
        if IsMultiProcess:
            pool = mp.Pool(MultiProcess)
        for fn in base_file_list:
            if not IsMultiProcess:
                MergeSampleNorm_FromSortedH5f( fn,new_stride,new_step,new_sorted_path,more_actions_config )
            else:
                results = pool.apply_async(MergeSampleNorm_FromSortedH5f,( fn,new_stride,new_step,new_sorted_path,more_actions_config ))
        if IsMultiProcess:
            pool.close()
            pool.join()

            success_fns = []
            success_N = len(base_file_list)
            try:
                for k in range(success_N):
                    success_fns.append(results.get(timeout=0.1))
            except:
                assert len(success_fns)==success_N,"MergeSampleNorm failed. only %d files successed"%(len(success_fns))
            print("\n\n MergeSampleNorm:all %d files successed\n******************************\n"%(len(success_fns)))

    def Sample(self,base_stride,base_step,numpoint_block,MultiProcess=0):
        base_path = self.house_h5f_dir+'/'+get_stride_step_name(base_stride,base_step)

        base_file_list = glob.glob( os.path.join(base_path,'*.sh5') )
        print('%d sh5 files in %s'%(len(base_file_list),base_path))

        IsGenNorm = False

        IsMultiProcess = MultiProcess>1
        if IsMultiProcess:
            pool = mp.Pool(MultiProcess)
        for fn in base_file_list:
            if not IsMultiProcess:
                SampleFile(fn,numpoint_block,IsGenNorm)
            else:
                results = pool.apply_async( SampleFile,(fn,numpoint_block,IsGenNorm) )
        if IsMultiProcess:
            pool.close()
            pool.join()

            success_fns = []
            success_N = len(base_file_list)
            try:
                for k in range(success_N):
                    success_fns.append(results.get(timeout=0.1))
            except:
                assert len(success_fns)==success_N,"Sample failed. only %d files successed"%(len(success_fns))
            print("\n\n SampleFile:all %d files successed\n******************************\n"%(len(success_fns)))

    def get_mean_pr_sample_num(self,base_stride,base_step):
        base_sorted_path = self.house_h5f_dir+'/'+get_stride_step_name(base_stride,base_step)
        py_normed_path = base_sorted_path +'_pyramid-'+GlobalSubBaseBLOCK.get_pyramid_flag()
        file_list = glob.glob( os.path.join(py_normed_path,'*.prh5') )
        sum_sg_bidxmap_sample_num = np.zeros((GlobalSubBaseBLOCK.cascade_num,6)).astype(np.uint64)
        sum_flatten_bmap_sample_num = np.zeros((GlobalSubBaseBLOCK.cascade_num,3)).astype(np.uint64)
        for fn in file_list:
            with h5py.File(fn,'r') as f:
                #print('sum_sg_bidxmap_sample_num')
                #print(f['bidxmaps'].attrs['sum_sg_bidxmap_sample_num'].astype(np.uint64))
                sum_sg_bidxmap_sample_num += f['bidxmaps'].attrs['sum_sg_bidxmap_sample_num'].astype(np.uint64)
                sum_flatten_bmap_sample_num += f['bidxmaps'].attrs['sum_flatten_bmap_sample_num'].astype(np.uint64)
        mean_sg_sample_num = get_mean_sg_sample_rate(sum_sg_bidxmap_sample_num)
        mean_flatten_sample_num = get_mean_flatten_sample_rate(sum_flatten_bmap_sample_num)
        print('mean_sg_sample_num:')
        print(mean_sg_sample_num)
        print('\nmean_flatten_sample_num')
        print(mean_flatten_sample_num)

    def GenPyramid(self,base_stride,base_step,MultiProcess=0):
      #  base_stride = [2,2,-1]
      #  base_step = [4,4,-1]
        base_sorted_path = self.house_h5f_dir+'/'+get_stride_step_name(base_stride,base_step)
        file_list = glob.glob( os.path.join(base_sorted_path,'*.sh5') )

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
            print("\n\n Norm:all %d files successed\n******************************\n"%(len(success_fns)))

    def Norm(self,base_stride,base_step,numpoint_block,MultiProcess=0):
      #  base_stride = [2,2,-1]
      #  base_step = [4,4,-1]
      #  numpoint_block = 8192
        base_sorted_sampled_path = self.house_h5f_dir+'/'+get_stride_step_name(base_stride,base_step)+'_'+str(numpoint_block)
        file_list = glob.glob( os.path.join(base_sorted_sampled_path,'*.rsh5') )


        IsMultiProcess = MultiProcess>1
        if IsMultiProcess:
            pool = mp.Pool(MultiProcess)
        for fn in file_list:
            if not IsMultiProcess:
                NormSortedSampledFlie(fn)
            else:
                results = pool.apply_async(NormSortedSampledFlie,(fn,))
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
            print("\n\n Norm:all %d files successed\n******************************\n"%(len(success_fns)))

    def MergeNormed(self,stride,step,numpoint_block,format, flag = 'region'):
        scans_name_ = self.scans_name.replace('/','_')[1:]
        if format == '.nh5':
            base_sorted_sampled_normed_path = self.house_h5f_dir+'/'+get_stride_step_name(stride,step)+'_'+str(numpoint_block)+'_normed'
            file_list = glob.glob( os.path.join(base_sorted_sampled_normed_path,'*.nh5') )
            config_flag = os.path.basename( base_sorted_sampled_normed_path )
        elif format == '.prh5':
            base_sorted_path = self.house_h5f_dir+'/'+get_stride_step_name(stride,step)
            py_normed_path = base_sorted_path +'_pyramid-'+GlobalSubBaseBLOCK.get_pyramid_flag()
            config_flag = os.path.basename( py_normed_path )
            merged_base_path = self.matterport3D_h5f_dir+'/'+config_flag
            merged_houses_path = merged_base_path+'/all_houses/'
            if flag == 'region':
                file_list = glob.glob( os.path.join(py_normed_path,'*.prh5') )
                merged_path = merged_houses_path
                merged_file_name = merged_path + self.house_name+format
                if not os.path.exists(merged_path):
                    os.makedirs(merged_path)
                MergeNormed_H5f(file_list,merged_file_name,IsShowSummaryFinished=True)

            elif flag == 'house':
                all_file_list = glob.glob( os.path.join(merged_houses_path,'*.prh5') )
                all_housenames_ls = [ os.path.splitext( os.path.basename(fn) )[0]  for fn in all_file_list]
                house_names_test = ['17DRP5sb8fy','1pXnuDYAj8r']
                train_basefn_ls = [ fn for fn in all_housenames_ls if not fn in house_names_test ]
                test_basefn_ls = [ fn for fn in all_housenames_ls if  fn in house_names_test ]
                train_basefn_ls.sort()
                test_basefn_ls.sort()
                train_file_ls = [ os.path.join(merged_houses_path,basefn+'.prh5') for basefn in train_basefn_ls ]
                test_file_ls = [ os.path.join(merged_houses_path,basefn+'.prh5') for basefn in test_basefn_ls ]

                train_merged_basefn = ''
                for i in range(len(train_basefn_ls)):
                    train_merged_basefn += train_basefn_ls[i][0:3]
                    if i < len(train_basefn_ls)-1:
                        train_merged_basefn += '-'
                test_merged_basefn = ''
                for i in range(len(test_basefn_ls)):
                    test_merged_basefn += test_basefn_ls[i][0:3]
                    if i < len(test_basefn_ls)-1:
                        test_merged_basefn += '-'

                merged_path = merged_base_path
                if not os.path.exists(merged_path):
                    os.makedirs(merged_path)
                train_merged_fn = os.path.join( merged_path, train_merged_basefn+'.prh5' )
                test_merged_fn = os.path.join( merged_path, test_merged_basefn+'.prh5' )

                MergeNormed_H5f(train_file_ls,train_merged_fn,IsShowSummaryFinished=True)
                MergeNormed_H5f(test_file_ls,test_merged_fn,IsShowSummaryFinished=True)



    def GenObj_RawH5f(self):
        file_name = self.house_rawh5f_dir+'/region1.rh5'
        file_name = '/DS/Matterport3D/Matterport3D_H5F/v1/scans/YFuZgdQ5vWj/rawh5f/region18.rh5'
        xyz_cut_rate= [0,0,0.9]
        with h5py.File(file_name,'r') as h5f:
            rawh5f = Raw_H5f(h5f,file_name)
            rawh5f.generate_objfile(IsLabelColor=False,xyz_cut_rate=xyz_cut_rate)

    def GenObj_SortedH5f(self):
        stride = step = [0.1,0.1,0.1]
        sorted_path = self.house_h5f_dir+'/'+get_stride_step_name(stride,step)
        file_name = sorted_path + '/region7.sh5'
        with h5py.File(file_name,'r') as h5f:
            sortedh5f = Sorted_H5f(h5f,file_name)
            sortedh5f.gen_file_obj(IsLabelColor=False)

    def GenObj_NormedH5f(self):
        stride = step = [0.1,0.1,0.1]
        file_name = self.house_h5f_dir+'/'+get_stride_step_name(step,stride) +'_pyramid-'+GlobalSubBaseBLOCK.get_pyramid_flag() + '/region5.prh5'
        with h5py.File(file_name,'r') as h5f:
            normedh5f = Normed_H5f(h5f,file_name)
            normedh5f.gen_gt_pred_obj_examples(['ALL'])
            #normedh5f.gen_gt_pred_obj_examples(['void'])

    def ShowSummary(self):
        file_name = self.house_rawh5f_dir+'/region1.rh5'
        step = stride = [0.1,0.1,0.1]
        file_name = self.house_h5f_dir+'/'+get_stride_step_name(step,stride) + '/region2.sh5'
        #file_name = self.house_rawh5f_dir + '/region2.rh5'
        #file_name = self.house_h5f_dir+'/'+get_stride_step_name(step,stride) +'_pyramid-'+GlobalSubBaseBLOCK.get_pyramid_flag() + '/region2.prh5'
        file_name = '/DS/Matterport3D/Matterport3D_H5F/v1/scans/17DRP5sb8fy/stride_0d1_step_0d1_pyramid-1_2-512_256_64-128_12_6-0d2_0d6_1d1/region1.prh5'
        #file_name = '/DS/Matterport3D/Matterport3D_H5F/all_merged_nf5/17DRP5sb8fy_stride_0d1_step_0d1_pyramid-1_2-512_256_64-128_12_6-0d2_0d6_1d1.prh5'
        IsIntact,check_str = check_h5fs_intact(file_name)
        if IsIntact:
            with h5py.File(file_name,'r') as h5f:
                show_h5f_summary_info(h5f)
        else:
            print("file not intact: %s \n\t %s"%(file_name,check_str))

    def ShowBidxmap(self):
        file_name = self.house_rawh5f_dir+'/region1.rh5'
        step = stride = [0.1,0.1,0.1]
        file_name = self.house_h5f_dir+'/'+get_stride_step_name(step,stride) + '/region2.sh5'
        with h5py.File(file_name,'r') as h5f:
            GlobalSubBaseBLOCK.show_all(h5f,file_name)


def parse_house(house_name = '17DRP5sb8fy',scans_name = '/v1/scans'):
    MultiProcess = 4
    matterport3d_prepare = Matterport3D_Prepare(house_name,scans_name)

    operations = ['ParseRaw','SortRaw','GenPyramid','MergeSampleNorm','Sample','Norm','MergeNormed']
    operations  = ['ParseRaw']
    operations  = ['SortRaw']
    operations  = ['GenPyramid']
    #operations  = ['GenPyramid','GenObj_NormedH5f']
    #operations  = ['MergeNormed-region']
    #operations  = ['MergeNormed-house']
    #operations  = ['GenObj_SortedH5f']
    #operations  = ['GenObj_RawH5f']
    #operations  = ['GenObj_NormedH5f']
    #operations  = ['pr_sample_rate']
    if 'ParseRaw' in operations:
        matterport3d_prepare.Parse_house_regions(MultiProcess)

    base_step_stride = [0.1,0.1,0.1]
    if 'SortRaw' in operations:
        matterport3d_prepare.SortRaw(base_step_stride,MultiProcess)

    if 'GenPyramid' in operations:
        matterport3d_prepare.GenPyramid(base_step_stride,base_step_stride,MultiProcess)
    if 'pr_sample_rate' in operations:
        matterport3d_prepare.get_mean_pr_sample_num(base_step_stride,base_step_stride)

    new_stride = [1,1,-1]
    new_step = [2,2,-1]
    numpoint_block = 8192

    new_stride=new_step=base_step_stride
    numpoint_block = 8
    if 'MergeSampleNorm' in operations:
        matterport3d_prepare.MergeSampleNorm(base_step_stride,new_stride,new_step,numpoint_block,MultiProcess)
    if 'Sample' in operations:
        new_stride=new_step=base_step_stride
        matterport3d_prepare.Sample(new_stride,new_step,numpoint_block,MultiProcess)
    if 'Norm' in operations:
        matterport3d_prepare.Norm(new_stride,new_step,numpoint_block,MultiProcess)
    if 'MergeNormed-region' in operations:
        matterport3d_prepare.MergeNormed(new_stride,new_step,numpoint_block,'.prh5','region')
    if 'MergeNormed-house' in operations:
        matterport3d_prepare.MergeNormed(new_stride,new_step,numpoint_block,'.prh5','house')
    if 'GenObj_RawH5f' in operations:
        matterport3d_prepare.GenObj_RawH5f()
    if 'GenObj_SortedH5f' in operations:
        matterport3d_prepare.GenObj_SortedH5f()
    if 'GenObj_NormedH5f' in operations:
        matterport3d_prepare.GenObj_NormedH5f()

def parse_house_ls():
    scans_name = '/v1/scans'
    house_names = ['17DRP5sb8fy']
    house_names_test = ['17DRP5sb8fy','1pXnuDYAj8r']
    house_names_train = ['17DRP5sb8fy','1pXnuDYAj8r','2azQ1b91cZZ','2t7WUuJeko7']
    house_names_train += ['5q7pvUzZiYa', '759xd9YjKW5','8194nk5LbLH','8WUmhLawc2A','ac26ZMwG7aT','B6ByNegPMKs']

    scans_name_abs = Matterport3D_Prepare.matterport3D_root_dir + scans_name
    all_house_names = os.listdir(scans_name_abs)

    house_names = all_house_names

    for house_name in house_names:
        parse_house(house_name,scans_name)

def show_summary():
    matterport3d_prepare = Matterport3D_Prepare()
    matterport3d_prepare.ShowSummary()
def show_bidxmap():
    matterport3d_prepare = Matterport3D_Prepare()
    matterport3d_prepare.ShowBidxmap()

def show_all_label_colors():
    Normed_H5f.show_all_colors('MATTERPORT')

if __name__ == '__main__':
    parse_house_ls()
    #show_summary()
    #show_bidxmap()
    #show_all_label_colors()


