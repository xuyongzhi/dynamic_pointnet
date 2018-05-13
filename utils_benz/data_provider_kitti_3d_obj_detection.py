# xyz Dec 2017


from __future__ import print_function
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
#from plyfile import (PlyData, PlyElement, make2d, PlyParseError, PlyProperty)
import numpy as np
import h5py
import glob
import time
import multiprocessing as mp
import itertools
from block_data_prep_util import Normed_H5f,Sorted_H5f,GlobalSubBaseBLOCK
from ply_util import create_ply
from utils_voxelnet import label_to_gt_box3d


ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR,'data')
DATASET_DIR={}
DATASET_DIR['scannet'] = os.path.join(DATA_DIR,'ScannetH5F')
DATASET_DIR['stanford_indoor3d'] = os.path.join(DATA_DIR,'stanford_indoor3d')
matterport3D_h5f_dir = os.path.join(DATA_DIR,'Matterport3D_H5F')
DATASET_DIR['matterport3d'] = matterport3D_h5f_dir
DATASET_DIR['Voxel'] = os.path.join(DATA_DIR,'Voxel')


DEBUGTMP = True
READING_RAW_LABEL = True   ## reading raw labeling data or pre-calculated data


#-------------------------------------------------------------------------------
# provider for training and testing
#------------------------------------------------------------------------------
class Net_Provider_kitti():          ## benz_m
    '''
    (1) provide data for training
    dataset_name: 'stanford_indoor3d' 'scannet'
    all_filename_glob:  stride_1_step_2_test_small_4096_normed/*.sph5
    eval_fnglob_or_rate: file name str glob or file number rate. 'scan1*.sph5' 0.2
    num_point_block: if the block point number is not this, do randomly sample
    feed_data_elements: sub list of ['xyz_1norm','xyz_midnorm','nxnynz','color_1norm','intensity_1norm']
    feed_label_elements: sub list of ['label_category','label_instance','label_material']
    '''
    # input normalized h5f files
    # normed_h5f['data']: [blocks*block_num_point*num_channel],like [1000*4096*9]
    # one batch would contain sevel(batch_size) blocks,this will be set out side
    # provider with train_start_idx and test_start_idx


    def __init__(self, net_configs, dataset_name,all_filename_glob,eval_fnglob_or_rate, bxmh5_folder_name,\
                 only_evaluate,feed_data_elements, feed_label_elements, num_point_block = None, \
                 train_num_block_rate=1,eval_num_block_rate=1 ):
        t_init0 = time.time()
        self.net_configs = net_configs
        self.bxmh5_folder_name = bxmh5_folder_name
        self.dataset_name = dataset_name
        self.feed_data_elements = feed_data_elements
        self.feed_label_elements = feed_label_elements
        self.num_point_block = num_point_block
        all_file_list = self.get_all_file_name_list(dataset_name,all_filename_glob)
        train_file_list,eval_file_list = self.split_train_eval_file_list\
                            (all_file_list,eval_fnglob_or_rate)
        if only_evaluate:
            open_type = 'a' # need to write pred labels
        else:
            open_type = 'r'
        train_file_list, train_bxmh5_fls, train_label_fn_ls = self.get_bxmh5_fn_ls( train_file_list )
        eval_file_list, eval_bxmh5_fls, eval_label_fn_ls = self.get_bxmh5_fn_ls( eval_file_list )
        self.train_file_N = train_file_N = len(train_file_list)
        self.eval_file_N = eval_file_N = len(eval_file_list)
        self.g_file_N = train_file_N + eval_file_N
        self.sph5_file_list =  sph5_file_list = train_file_list + eval_file_list
        self.bxmh5_fn_ls = train_bxmh5_fls + eval_bxmh5_fls
        self.labels_fn_ls = labels_fn_ls  = train_label_fn_ls + eval_label_fn_ls
        if len(sph5_file_list) > 6:
            print('WARING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\ntoo many (%d) files can lead to long read time'%(len(sph5_file_list)))
        #-----------------------------------------------------------------------
        # open each file as a Normed_H5f class instance
        self.norm_h5f_L = []
        self.label_data_all = []
        # self.g_block_idxs: within the whole train/test dataset  (several files)
        #     record the start/end row idx  of each file to help search data from all files
        #     [ [start_global_row_idxs,end_global__idxs] ]
        #     [[  0,  38], [ 38,  90],[ 90, 150],...[259, 303],[303, 361],[361, 387]]
        #   self.train_num_blocks: 303
        #   self.eval_num_blocks: 84
        #   self.eval_global_start_idx: 303
        self.g_block_idxs = np.zeros((self.g_file_N,2),np.int32)
        self.eval_global_start_idx = None

        file_N = len(sph5_file_list)
        print('Totally %d files. \nReading the block number in each file.'%(file_N))
        t_start_global_blockid = time.time()
        for i,fn in enumerate(sph5_file_list):
            assert(os.path.exists(fn))

            sph5_file_name = os.path.basename(fn).replace(".sph5","")
            h5f = h5py.File(fn,open_type)
            norm_h5f = Normed_H5f(h5f,fn)
            self.norm_h5f_L.append( norm_h5f )

            label_file_name = os.path.basename(labels_fn_ls[i]).replace("_label.txt","")
            assert sph5_file_name == label_file_name
            label_data_i = self.read_label_data( labels_fn_ls[i] )  ## benz_m, opening all the bounding box
            self.label_data_all.append( label_data_i )

            file_block_N = self.get_block_n(norm_h5f)
            self.g_block_idxs[i,1] = self.g_block_idxs[i,0] + file_block_N
            if i<self.g_file_N-1:
                self.g_block_idxs[i+1,0] = self.g_block_idxs[i,1]

        t_end_global_blockid = time.time()
        print('global block id reading finished, t=%f ms'%(1000.0*(t_end_global_blockid-t_start_global_blockid)))

        if train_file_N < file_N:
            self.eval_global_start_idx = self.g_block_idxs[train_file_N,0]
        else:
            self.eval_global_start_idx = self.g_block_idxs[-1,1]
        if train_file_N > 0:
            self.train_num_blocks = self.g_block_idxs[train_file_N-1,1] # = self.eval_global_start_idx
        else: self.train_num_blocks = 0
        self.eval_num_blocks = self.g_block_idxs[-1,1] - self.train_num_blocks
        self.num_classes = self.norm_h5f_L[0].num_classes
        self.label_ele_idxs_isph5f = self.norm_h5f_L[0].label_ele_idxs
        self.label_eles_isph5f = self.norm_h5f_L[0].label_set_elements

        # self.update_sample_loss_weight()   # benz_m, label is not available
        t_end_update_loss_weight = time.time()
        print('update_sample_loss_weight t: %f ms'%(1000*(t_end_update_loss_weight-t_end_global_blockid)))
        self.update_train_eval_shuffled_idx()

        #-----------------------------------------------------------------------
        # use only part of the data to test code:
        if train_num_block_rate!=1 or eval_num_block_rate!=1:
            self.get_data_label_shape()
            print('whole train data shape: %s'%(str(self.whole_train_data_shape)))
            print('whole eval data shape: %s'%(str(self.whole_eval_data_shape)))
            # train: use the front part
            self.train_num_blocks = int( self.train_num_blocks * train_num_block_rate )
            if not only_evaluate:
                self.train_num_blocks = max(self.train_num_blocks,2)
            new_eval_num_blocks = int( max(2,self.eval_num_blocks * eval_num_block_rate) )
            # eval:use the back part, so train_file_list and eval_file_list can be
            # the same
            self.eval_global_start_idx += self.eval_num_blocks - new_eval_num_blocks
            self.eval_num_blocks = new_eval_num_blocks

        self.gsbb_load = GlobalSubBaseBLOCK(  )
        self.gsbb_load.load_para_from_bxmh5( self.bxmh5_fn_ls[0] )
        self.global_num_point = self.gsbb_load.global_num_point
        self.gsbb_config = self.gsbb_load.gsbb_config
        self.get_data_label_shape_info()
        t_get_data_laebl_shape = time.time()
        print('get_data_label_shape t: %f ms'%(1000*(t_get_data_laebl_shape - t_end_update_loss_weight)))
        self.update_data_summary()

        print('Net_Provider init t: %f ms\n\n'%(1000*(time.time()-t_init0)))

    def get_bxmh5_fn_ls( self, plsph5_fn_ls ):
        bxmh5_fn_ls = []
        plsph5_fn_ls_new  = []
        label_fn_ls = []
        for plsph5_fn in plsph5_fn_ls:
            bxmh5_fn = self.get_bxmh5_fn_1( plsph5_fn )
            label_fn = self.get_label_fn_1(plsph5_fn)
            sph5_name  = os.path.basename(plsph5_fn).replace(".sph5","")   ## benz_m, checking
            bxmh5_name = os.path.basename(bxmh5_fn).replace(".bxmh5","")
            label_name = os.path.basename(label_fn).replace("_label.txt","")
            assert sph5_name==bxmh5_name
            assert bxmh5_name==label_name
            if os.path.exists( bxmh5_fn ):
                # check shapes match with each other
                with h5py.File( bxmh5_fn, 'r' ) as bxmh5f:
                    with h5py.File( plsph5_fn, 'r' ) as plsph5f:
                        with open(label_fn, 'r') as label_txt:
                            if bxmh5f['bidxmaps_sample_group'].shape[0] == plsph5f['data'].shape[0] and bxmh5f['bidxmaps_sample_group'].shape[0]==len(label_txt.readlines()) :  ## benz_m , double checking the number of label is similar with that of point data
                                bxmh5_fn_ls.append( bxmh5_fn )
                                plsph5_fn_ls_new.append( plsph5_fn )
                                label_fn_ls.append(label_fn)
                            else:
                                print('bxmh5(%d) and plsph5(%d) shapes do not match for %s'%( bxmh5f['bidxmaps_flat'].shape[0], plsph5f['data'].shape[0],plsph5_fn ))
                                assert False
            else:
                print( 'not exist: %s'%(bxmh5_fn) )
        return plsph5_fn_ls_new, bxmh5_fn_ls, label_fn_ls

    def get_bxmh5_fn( self, plsph5_fn ):
        house_name = os.path.splitext( os.path.basename(plsph5_fn) )[0]
        pl_config_dir_name = os.path.dirname(plsph5_fn)
        pl_config_name = os.path.basename(pl_config_dir_name)
        each_house_dirname = os.path.dirname( pl_config_dir_name )
        bxmh5_dirname = os.path.join( each_house_dirname, self.bxmh5_folder_name )
        bxmh5_fn = os.path.join( bxmh5_dirname, house_name + '.bxmh5' )
        return bxmh5_fn

    def get_bxmh5_fn_1( self, plsph5_fn ):
        #fn1 = os.path.dirname(os.path.dirname( os.path.dirname(plsph5_fn) ))
        #tmp = plsph5_fn.split( os.sep )
        #tmp[-2] = self.bxmh5_folder_name
        ##tmp[-3] = self.bxmh5_folder_name
        #base_fn = os.path.splitext('/'+os.path.join( *tmp ))[0]
        #bxmh5_fn = base_fn+'.bxmh5'

        path = os.path.dirname(os.path.dirname( os.path.dirname(plsph5_fn) )) +'/'+ self.bxmh5_folder_name + '/'
        base_name = os.path.splitext( os.path.basename( plsph5_fn ) )[0]
        bxmh5_fn = path + base_name + '.bxmh5'
        return bxmh5_fn

    def get_label_fn_1( self, plsph5_fn ):

        path = os.path.dirname(os.path.dirname( os.path.dirname(plsph5_fn) )) +'/'+ self.bxmh5_folder_name + '/'
        base_name = os.path.splitext( os.path.basename( plsph5_fn ) )[0]
        bxmh5_fn = path + base_name + '_label' + '.txt'
        return bxmh5_fn



    def get_data_label_shape_info(self):
        self.feed_data_ele_idxs,self.feed_label_ele_idxs = self.norm_h5f_L[0].get_feed_ele_ids(self.feed_data_elements,self.feed_label_elements)
        self.data_num_eles = len([idx for e in self.feed_data_ele_idxs for idx in self.feed_data_ele_idxs[e] ])
        self.label_num_eles = len([self.feed_label_ele_idxs[e] for e in self.feed_label_ele_idxs])
        self.cascade_num = self.gsbb_load.cascade_num
        self.whole_train_data_shape = np.array([ self.train_num_blocks, self.gsbb_load.global_num_point, self.data_num_eles])
        self.whole_eval_data_shape = np.copy(self.whole_train_data_shape)
        self.whole_eval_data_shape[0] = self.eval_num_blocks
        self.whole_train_label_shape = np.copy(self.whole_train_data_shape)
        self.whole_train_label_shape[-1] = self.label_num_eles
        print('\ntrain data shape',self.whole_train_data_shape)
        print('train label shape',self.whole_train_label_shape)
        print('eval data shape',self.whole_eval_data_shape)

        #self.get_data_label_shape_byread()

        block_sample = self.whole_train_data_shape[1:-1]
        if block_sample.size==1:
            block_sample= (block_sample[0],)
        elif block_sample.size==2:
            block_sample = (block_sample[0],block_sample[1])
        self.block_sample = block_sample

        self.sg_bidxmaps_shape = self.gsbb_load.get_sg_bidxmaps_fixed_shape()
        self.flatten_bidxmaps_shape = self.gsbb_load.get_flatten_bidxmaps_shape()
        self.sg_bidxmaps_extract_idx = self.gsbb_load.sg_bidxmaps_extract_idx
        self.flatten_bidxmaps_extract_idx = self.gsbb_load.flatten_bidxmaps_extract_idx
        print('sg_bidxmaps_shape',self.sg_bidxmaps_shape)
        print('flatten_bidxmaps_shape',self.flatten_bidxmaps_shape)
        print('sg_bidxmaps_extract_idx\n',self.sg_bidxmaps_extract_idx)
        print('flatten_bidxmaps_extract_idx\n',self.flatten_bidxmaps_extract_idx)

    def get_data_label_shape_byread(self):
        t0 = time.time()
        data_batches,label_batches,sample_weights,sg_bidxmaps,flatten_bidxmaps, fmap_neighbor_idises = self.get_train_batch(0,min(self.train_num_blocks,32))
        #data_batches,label_batches,_ = self.get_train_batch(0,1)
        self.whole_train_data_shape = np.array(data_batches.shape)
        self.whole_train_data_shape[0] = self.train_num_blocks
        self.data_num_eles = self.whole_train_data_shape[-1]
        self.whole_eval_data_shape = np.array(data_batches.shape)
        self.whole_eval_data_shape[0] = self.eval_num_blocks
        self.whole_train_label_shape = np.array(label_batches.shape)
        self.whole_train_label_shape[0] = self.train_num_blocks
        self.label_num_eles = label_batches.shape[-1]
        print('\ntrain data shape',self.whole_train_data_shape)
        print('train label shape',self.whole_train_label_shape)
        print('eval data shape',self.whole_eval_data_shape)
        print('read %d global block t: %f ms\n'%( data_batches.shape[0], 1000*(time.time()-t0)))

    def check_bidxmap(self):
        datas,labels,sample_weights,sg_bidxmaps,flatten_bidxmaps, fmap_neighbor_idises = self.get_train_batch(0,min(self.train_num_blocks,32))


    def get_block_n(self,norm_h5f):
        file_block_N = norm_h5f.data_set.shape[0]
            #file_block_N = GlobalSubBaseBLOCK.get_block_n_of_new_stride_step_(norm_h5f.h5f,norm_h5f.file_name,'global')
        return file_block_N

    def update_data_summary(self):
        self.data_summary_str = '%s \nfeed_data_elements:%s \nfeed_label_elements:%s \n'%(self.dataset_name,self.feed_data_elements,self.feed_label_elements)
        self.data_summary_str += 'train data shape: %s \ntest data shape: %s \n'%(
                         str(self.whole_train_data_shape),str(self.whole_eval_data_shape))
        ## benz_m, the label_histgram is not available in the object detection
        # self.data_summary_str += 'train labels histogram: %s \n'%( np.array_str(np.transpose(self.train_labels_hist_1norm) ))
        # self.data_summary_str += 'test labels histogram: %s \n'%( np.array_str(np.transpose(self.test_labels_hist_1norm) ))
        # self.data_summary_str += 'label histogram:'+','.join('%5.2f'%(lh) for lh in self.labels_hist_1norm[:,0].tolist()) + '\n'
        # self.data_summary_str += 'label   weights:'+','.join('%5.2f'%(lw) for lw in self.labels_weights[:,0].tolist()) + '\n'
        # self.data_summary_str += 'class      name:'+','.join( '%5s'%(Normed_H5f.g_label2class_dic[self.dataset_name][label][0:5])  for label in range(len(Normed_H5f.g_label2class_dic[self.dataset_name])) )
        # print(self.data_summary_str)

    def get_all_file_name_list(self,dataset_name,all_filename_globs):
        all_file_list = []
        fn_globs = []
        file_format = '.sph5'

        for all_filename_glob in all_filename_globs:
            fn_glob = os.path.join(DATASET_DIR[dataset_name],all_filename_glob+'*'+file_format)
            fn_candis  = glob.glob( fn_glob )
            for fn in fn_candis:
                IsIntact,_ = Normed_H5f.check_sph5_intact( fn )
                if IsIntact:
                    all_file_list.append(fn)

            fn_globs.append(fn_glob)
        assert len(all_file_list)!=0,"no file in %s"%(fn_globs)
        #assert len(all_file_list)!=1,"only one file, should > 1 to seperate as train and test"
        return all_file_list

    def split_train_eval_file_list(self,all_file_list,eval_fnglob_or_rate=None):
        if eval_fnglob_or_rate == None:
            if self.dataset_name=='stanford_indoor3d':
                eval_fnglob_or_rate = 'Area_6'
            if self.dataset_name=='scannet':
                eval_fnglob_or_rate = 0.2
        if type(eval_fnglob_or_rate)==str:
            # split by name
            train_file_list = []
            eval_file_list = []
            for fn in all_file_list:
                if fn.find(eval_fnglob_or_rate) > 0:
                    eval_file_list.append(fn)
                else:
                    train_file_list.append(fn)
        elif type(eval_fnglob_or_rate) == float:
            if eval_fnglob_or_rate<0:
                inverse = True
                eval_fnglob_or_rate=-eval_fnglob_or_rate
            else:
                inverse = False
            all_file_list.sort()
            # split by number
            n = len(all_file_list)
            m = int(n*(1-eval_fnglob_or_rate))
            if not inverse:
                train_file_list = all_file_list[0:m]
                eval_file_list = all_file_list[m:n]
            else:
                eval_file_list = all_file_list[0:m]
                train_file_list = all_file_list[m:n]

        if len(eval_file_list) == 0 and eval_fnglob_or_rate<0:
            eval_file_list = [train_file_list[0]]

        if len(eval_file_list) == 0:    ## benz_m
            eval_file_list = [train_file_list[1]]



        log_str = '\ntrain file list (n=%d) = \n%s\n\n'%(len(train_file_list),train_file_list[-2:])
        log_str += 'eval file list (n=%d) = \n%s\n\n'%(len(eval_file_list),eval_file_list[-2:])
        print( log_str )
        return train_file_list, eval_file_list


    def test_tmp(self):
        s = 0
        e = 1
        train_data,train_label = self.get_train_batch(s,e)
        eval_data,eval_label = self.get_eval_batch(s,e)
        print('train:\n',train_data[0,0,:])
        print('eval:\n',eval_data[0,0,:])
        print('err=\n',train_data[0,0,:]-eval_data[0,0,:])


    def __exit__(self):
        print('exit Net_Provider')
        for norm_h5f in self.norm_h5f:
            norm_h5f.h5f.close()

    def global_idx_to_local(self,g_start_idx,g_end_idx):
        assert(g_start_idx>=0 and g_start_idx<=self.g_block_idxs[-1,1])
        assert(g_end_idx>=0 and g_end_idx<=self.g_block_idxs[-1,1])
        for i in range(self.g_file_N):
            if g_start_idx >= self.g_block_idxs[i,0] and g_start_idx < self.g_block_idxs[i,1]:
                start_file_idx = i
                local_start_idx = g_start_idx - self.g_block_idxs[i,0]
                for j in range(i,self.g_file_N):
                    if g_end_idx > self.g_block_idxs[j,0] and g_end_idx <= self.g_block_idxs[j,1]:
                        end_file_idx = j
                        local_end_idx = g_end_idx - self.g_block_idxs[j,0]

        return start_file_idx,end_file_idx,local_start_idx,local_end_idx

    def set_pred_label_batch(self,pred_label,g_start_idx,g_end_idx):
        start_file_idx,end_file_idx,local_start_idx,local_end_idx = \
            self.global_idx_to_local(g_start_idx,g_end_idx)
        pred_start_idx = 0
        for f_idx in range(start_file_idx,end_file_idx+1):
            if f_idx == start_file_idx:
                start = local_start_idx
            else:
                start = 0
            if f_idx == end_file_idx:
                end = local_end_idx
            else:
                end = self.get_block_n( self.norm_h5f_L[f_idx] )
            n = end-start
            self.norm_h5f_L[f_idx].set_dset_value('pred_label',\
                pred_label[pred_start_idx:pred_start_idx+n,:],start,end)
            pred_start_idx += n
            self.norm_h5f_L[f_idx].h5f.flush()


    @staticmethod
    def get_indices_in_voxel( sg_bidxmaps, sg_bidxmaps_extract_idx, sub_block_stride_candis, sub_block_step_candis ):
        '''
        Convert from block center xyz_mm to point indices inside voxel.
        This should be performed in training model, just test it here in advance.
        '''
        aimb_center_xyz_mm_ls = []
        strides_mm = sub_block_stride_candis * 1000
        steps_mm =  sub_block_step_candis * 1000
        points_indices_in_voxel_all = []
        for cascade_id in range( len(strides_mm) ):
            sg_bidxmap_acxm =  GlobalSubBaseBLOCK.extract_sg_bidxmaps_( sg_bidxmaps, sg_bidxmaps_extract_idx, cascade_id, flag='both' )
            sg_bidxmap = sg_bidxmap_acxm[...,0:sg_bidxmap_acxm.shape[-1]-3]
            aimb_center_xyz_mm = sg_bidxmap_acxm[...,-3:sg_bidxmap_acxm.shape[-1]]
            aimb_center_xyz_mm_ls.append( aimb_center_xyz_mm )
            if cascade_id>0:
                # get the grouped xyz
                assert sg_bidxmap.shape[0] == 1
                points_indices_in_voxel_ls = []
                for batch in range(sg_bidxmap.shape[0]):
                    grouped_points_xyz_mm = np.take( aimb_center_xyz_mm_ls[cascade_id-1][batch], sg_bidxmap[batch], axis=0 )
                    points_indices_ls = []
                    for aimb in range(grouped_points_xyz_mm.shape[0]):
                        # points_invoxel_xyzs_mm is positions of all the points inside a voxels
                        # point_stride_mm is the stride between these points
                        points_invoxel_xyzs_mm = grouped_points_xyz_mm[aimb,:]
                        voxel_center_xyz_mm = aimb_center_xyz_mm[batch,aimb]
                        min_point_xyz_mm = voxel_center_xyz_mm - steps_mm[cascade_id]*0.5 + steps_mm[cascade_id-1]*0.5 # [ 950, 1550, 1750] - 300 *0.5 + 100*0.5
                        points_indices = (points_invoxel_xyzs_mm - min_point_xyz_mm) * 1.0 / strides_mm[cascade_id-1] # [1050, 1650, 1850] - [ 850., 1450., 1650.] / 100
                        points_indices = np.rint( points_indices ).astype( np.int32 )
                        points_indices_ls.append(np.expand_dims(points_indices,axis=0))

                        max_indice = (steps_mm[cascade_id] - steps_mm[cascade_id-1]) / strides_mm[cascade_id-1]
                        assert points_indices.min() >= 0
                        assert points_indices.max() <= max_indice
                    points_indices_batch = np.concatenate( points_indices_ls, 0 )
                    points_indices_in_voxel_ls.append( np.expand_dims(points_indices_batch,0) )
                points_indices_in_voxe = np.concatenate( points_indices_in_voxel_ls, 0 )
                points_indices_in_voxel_all.append( points_indices_in_voxe )
        import pdb; pdb.set_trace()  # XXX BREAKPOINT
        pass


    def read_label_data( self, label_fn_ls ):     ## benz_m, opening all the bounding box in on merged file
        label_data_i = []
        with open(label_fn_ls, 'r') as label_txt:
            all_names = label_txt.read().split('\n')[:-1]
            if READING_RAW_LABEL:   ##
                labels = []
                bounding_box = []
                for name in all_names:
                    dir_label = DATASET_DIR['Voxel'] + '/raw_label/' + name
                    assert os.path.exists(dir_label)
                    label = np.array([line for line in open( dir_label, 'r').readlines()])  # (N')
                    bounding_box.append(label)
                    #with open(dir_label,'r' as f:
                    #    bounding_box = label_to_gt_box3d(labels, cls='Car', coordinate='lidar')
                assert bounding_box
                label_data_i = label_to_gt_box3d(bounding_box, cls='Car', coordinate = 'lidar')

            else:
                for name in all_names:
                    dir_label = DATASET_DIR['Voxel'] + '/label/' + name
                    assert os.path.exists(dir_label)
                    bounding_box = []
                    with open(dir_label,'r') as  f:
                        labels = f.read().split('\n')
                        for label in labels:
                            if not label:
                                continue
                            label = label.split(' ')
                            if label[0] == 'Car':
                                car_label = np.concatenate((np.array([1]),label[1:8]), axis = 0) ## 1 means car
                                # bounding_box.append(label[1:8])
                                bounding_box.append(car_label)
                    assert bounding_box
                    data = np.array(bounding_box, dtype = np.float32)
                    label_data_i.append(data)
        assert label_data_i
        return label_data_i

    def get_global_batch_kitti(self,g_start_idx,g_end_idx):
        start_file_idx,end_file_idx,local_start_idx,local_end_idx = \
            self.global_idx_to_local(g_start_idx,g_end_idx)
        #t0 = time.time()

        # max_num_label = 0   ## benz_m
        # data_ls = []
        # label_ls = []
        # sg_bidxmaps_ls = []
        fid_start_end = []

        f_idx = start_file_idx
        assert f_idx == start_file_idx
        start = local_start_idx
        assert f_idx == end_file_idx
        end = local_end_idx

        fid_start_end.append( np.expand_dims(np.array([f_idx, start, end]),0) )

        new_feed_data_elements = list( self.feed_data_elements )
        if 'xyz' not in new_feed_data_elements:
            new_feed_data_elements = ['xyz'] + new_feed_data_elements
        if 'xyz_midnorm_block' in new_feed_data_elements:
            del  new_feed_data_elements[ new_feed_data_elements.index('xyz_midnorm_block') ]
        if 'xyz_1norm_block' in new_feed_data_elements:
            del  new_feed_data_elements[ new_feed_data_elements.index('xyz_1norm_block') ]

        new_feed_data_ele_idxs,_ = self.norm_h5f_L[0].get_feed_ele_ids(new_feed_data_elements, self.feed_label_elements)
        # t0=time.time()
        # data_i = self.norm_h5f_L[f_idx].get_normed_data(start,end, new_feed_data_elements)
        data_i = self.norm_h5f_L[f_idx].get_data_byeles(start,end, new_feed_data_elements)

        # label_i = self.norm_h5f_L[f_idx].get_label_eles(start,end, self.feed_label_elements)
        # t1=time.time()
        label_i = self.label_data_all[f_idx][start]
        # num_temp = label_i.shape[0]
        # if max_num_label <= num_temp:
        #    max_num_label = num_temp
        # data_i: [batch_size,npoint_block,data_nchannels]
        # label_i: [batch_size,npoint_block,label_nchannels]
        # sg_bidxmaps, flatten_bidxmaps, fmap_neighbor_idises = Normed_H5f.get_bidxmaps( self.bxmh5_fn_ls[f_idx],start,end )
        # t2=time.time()
        sg_bidxmaps, globalb_bottom_center_xyz =Normed_H5f.get_bidxmaps( self.bxmh5_fn_ls[f_idx],start,end )
        # t3=time.time()

        # print('sph5 time:{}, label time: {}, bidxmpa time: {}'.format((t1-t0),(t2-t1),(t3-t2)))
        # assert data_i.ndim == label_i.ndim and (data_i.shape[0:-1] == label_i.shape[0:-1])


        return data_i, label_i, sg_bidxmaps, globalb_bottom_center_xyz, fid_start_end




    def get_global_batch(self,g_start_idx,g_end_idx):
        start_file_idx,end_file_idx,local_start_idx,local_end_idx = \
            self.global_idx_to_local(g_start_idx,g_end_idx)
        #t0 = time.time()

        max_num_label = 0   ## benz_m
        data_ls = []
        label_ls = []
        center_mask = []
        sg_bidxmaps_ls = []
        flatten_bidxmaps_ls = []
        fmap_neighbor_idis_ls = []
        fid_start_end = []
        xyz_mid_ls = []
        for f_idx in range(start_file_idx,end_file_idx+1):
            '''
            if f_idx == start_file_idx:
                start = local_start_idx
            else:
                start = 0
            if f_idx == end_file_idx:
                end = local_end_idx
            else:
                end = self.get_block_n(self.norm_h5f_L[f_idx])
            '''
            assert f_idx == start_file_idx
            start = local_start_idx
            assert f_idx == end_file_idx
            end = local_end_idx


            fid_start_end.append( np.expand_dims(np.array([f_idx, start, end]),0) )

            new_feed_data_elements = list( self.feed_data_elements )
            if 'xyz' not in new_feed_data_elements:
                new_feed_data_elements = ['xyz'] + new_feed_data_elements
            if 'xyz_midnorm_block' in new_feed_data_elements:
                del  new_feed_data_elements[ new_feed_data_elements.index('xyz_midnorm_block') ]
            if 'xyz_1norm_block' in new_feed_data_elements:
                del  new_feed_data_elements[ new_feed_data_elements.index('xyz_1norm_block') ]

            new_feed_data_ele_idxs,_ = self.norm_h5f_L[0].get_feed_ele_ids(new_feed_data_elements, self.feed_label_elements)
            data_i = self.norm_h5f_L[f_idx].get_normed_data(start,end, new_feed_data_elements)
            # label_i = self.norm_h5f_L[f_idx].get_label_eles(start,end, self.feed_label_elements)
            label_i = self.label_data_all[f_idx][start]
            num_temp = label_i.shape[0]
            if max_num_label <= num_temp:
                max_num_label = num_temp
            # data_i: [batch_size,npoint_block,data_nchannels]
            # label_i: [batch_size,npoint_block,label_nchannels]
            sg_bidxmaps, flatten_bidxmaps, fmap_neighbor_idises = Normed_H5f.get_bidxmaps( self.bxmh5_fn_ls[f_idx],start,end )

            assert data_i.ndim == label_i.ndim and (data_i.shape[0:-1] == label_i.shape[0:-1])

            # get xyz_mid
            xyz_i = data_i[..., new_feed_data_ele_idxs['xyz']]
            xyz_min = xyz_i.min( axis=1 )
            xyz_max = xyz_i.max( axis=1 )
            xyz_mid = (xyz_min + xyz_max)/2
            xyz_midnorm_block_i = xyz_i - np.expand_dims( xyz_mid,1 )
            xyz_mid_ls.append( xyz_mid )

            if 'xyz' not in self.feed_data_elements:
                data_i = np.delete( data_i, new_feed_data_ele_idxs['xyz'], 2 )
                si = 0
            else:
                si = 1
            if 'xyz_midnorm_block' in self.feed_data_elements:
                data_i = np.concatenate( [xyz_midnorm_block_i, data_i],2 )
                assert self.feed_data_elements.index( 'xyz_midnorm_block' ) == si
                si += 1
            if 'xyz_1norm_block' in self.feed_data_elements:
                xyz_1norm_block_i = xyz_midnorm_block_i / self.norm_h5f_L[f_idx].h5f.attrs['block_step']
                data_i = np.concatenate( [xyz_1norm_block_i, data_i],2 )
                assert self.feed_data_elements.index( 'xyz_1norm_block' ) == si
                si += 1
            assert 'xyz_1norm_file' not in self.feed_data_elements
            #if 'xyz_1norm_file' in self.feed_data_elements:
            #    xyz_1norm_file_i = xyz_i / self.norm_h5f_L[f_idx].h5f.attrs['xyz_scope_aligned']
            #    data_i = np.concatenate( [xyz_1norm_file_i, data_i],2 )
            #    assert self.feed_data_elements.index( 'xyz_1norm_block' ) == si
            #    si += 1

            data_ls.append(data_i)
            label_ls.append(label_i)
            sg_bidxmaps_ls.append(sg_bidxmaps)
            flatten_bidxmaps_ls.append(flatten_bidxmaps)
            fmap_neighbor_idis_ls.append(fmap_neighbor_idises )

            center_mask_i = self.get_center_mask(f_idx, xyz_midnorm_block_i)
            center_mask.append(center_mask_i)



        data_batches = np.concatenate(data_ls,0)
        label_batches = np.concatenate(label_ls,0)
        sg_bidxmaps = np.concatenate( sg_bidxmaps_ls,axis=0 )
        flatten_bidxmaps = np.concatenate( flatten_bidxmaps_ls,axis=0 )
        fmap_neighbor_idises = np.concatenate( fmap_neighbor_idis_ls,0 )

        xyz_mid_batches = np.concatenate( xyz_mid_ls, 0 )
        center_mask = np.concatenate(center_mask,0)

        # sampling again
        #if self.num_point_block!=None:
            #assert data_batches.shape[1] == self.num_point_block
            #data_batches,label_batches = self.sample(data_batches,label_batches,self.num_point_block)

        num_label_eles = len(self.feed_label_elements)
        center_mask = np.expand_dims(center_mask,axis=-1)
        center_mask = np.tile(center_mask,(1,1,num_label_eles))

        # for each label, there is a weight. For all weight, when the point is
        # at edge, the weight is set to 0
        if self.net_configs['loss_weight'] != 'E': # Equal
            sample_weights = []
            for k in range(num_label_eles):
                if k == self.feed_label_ele_idxs['label_category'][0]:
                    sample_weights_k = np.take(self.labels_weights[:,k],label_batches[...,k])
                else:
                    sample_weights_k = np.ones_like(label_batches[...,k])
                sample_weights.append( np.expand_dims(sample_weights_k,axis=-1) )
            sample_weights = np.concatenate(sample_weights,axis=-1).astype( np.float32 )

        if self.net_configs['loss_weight'] == 'E': # Equal
            sample_weights = np.ones_like(label_batches)
        elif self.net_configs['loss_weight'] == 'N': # Number
            sample_weights = sample_weights
        elif self.net_configs['loss_weight'] == 'C': # Center
            sample_weights = np.ones_like(sample_weights) * center_mask
        elif self.net_configs['loss_weight'] == 'CN':
            sample_weights = sample_weights * center_mask
        else:
            assert False

        fid_start_end = np.concatenate( fid_start_end,0 )

        #Net_Provider.get_indices_in_voxel( sg_bidxmaps, self.sg_bidxmaps_extract_idx, self.gsbb_load.sub_block_stride_candis, self.gsbb_load.sub_block_step_candis  )

     #   print('\nin global')
     #   print('file_start = ',start_file_idx)
     #   print('file_end = ',end_file_idx)
     #   print('local_start = ',local_start_idx)
     #   print('local end = ',local_end_idx)
     #   #print('data = \n',data_batches[0,:])
        #t_block = (time.time()-t0)/(g_end_idx-g_start_idx)
        #print('get_global_batch t_block:%f ms'%(1000.0*t_block))
       # print(sg_bidxmaps.shape)
       # print(flatten_bidxmaps.shape)

        return data_batches, label_batches, sample_weights, sg_bidxmaps, flatten_bidxmaps, fmap_neighbor_idises, fid_start_end, xyz_mid_batches

    def get_fn_from_fid(self,fid):
        return self.sph5_file_list[ fid ]

    def get_center_mask(self,f_idx, xyz_midnorm, edge_rate=0.1):
        # true for center, false for edge
        # edge_rate: distance to center of the block_step
        block_step = self.norm_h5f_L[f_idx].h5f.attrs['block_step']
        center_rate = np.abs(xyz_midnorm / block_step) # -0.5 ~ 0.5
        # NOTE: Always true along z direction
        center_mask = (center_rate[...,0] < (0.5-edge_rate)) * ( center_rate[...,1] < (0.5-edge_rate) )
        #print('center n rate= %f'%(np.sum(center_mask).astype(float)/xyz_midnorm.shape[0]/xyz_midnorm.shape[1]))
        return center_mask






    def get_shuffled_global_batch(self,g_shuffled_idx_ls):
        IsRecordTime = False
        if IsRecordTime: t0 = time.time()
        max_num_label = 0
        data_batches = []
        label_batches = []
        sample_weights = []
        sg_bidxmaps_ls = []
        flatten_bidxmaps_ls = []
        globalb_bottom_center_xyz_ls = []
        fmap_neighbor_idis_ls = []
        fid_start_end_ls = []
        xyz_mid_ls = []
        if IsRecordTime: t1 = time.time()
        for idx in g_shuffled_idx_ls:
            # data_i,label_i,smw_i,sg_bidxmaps_i,flatten_bidxmaps_i, fmap_neighbor_idis_i,fid_start_end_i, xyz_mid_i = self.get_global_batch_kitti(idx,idx+1)   ## benz_m
            data_i,label_i,sg_bidxmaps_i, globalb_bottom_center_xyz_i, fid_start_end_i = self.get_global_batch_kitti(idx, idx+1)   ## benz_m

            sg_bidxmaps_ls.append(sg_bidxmaps_i)
            # flatten_bidxmaps_ls.append(flatten_bidxmaps_i)  # benz_m
            # fmap_neighbor_idis_ls.append( fmap_neighbor_idis_i )
            data_batches.append(data_i)
            globalb_bottom_center_xyz_ls.append(globalb_bottom_center_xyz_i)

            #label_batches.append( label_i[:,[0,1,2,4,5,6]]) # benz_m, 0:category, 1:l, 2:w, 3:alpha, 4:x, 5:y, for birdview detection
            label_batches.append( label_i)
            # num_temp = label_i.shape[0]
            #if max_num_label <= num_temp:
            #    max_num_label = num_temp

            # label_batches.append(label_i)
            # sample_weights.append(smw_i)
            fid_start_end_ls.append(fid_start_end_i)
            # xyz_mid_ls.append( xyz_mid_i )

        if IsRecordTime: t2 = time.time()
        data_batches = np.concatenate(data_batches,axis=0)
        # label_batches = np.concatenate(label_batches,axis=0)

        #labels_dims = label_batches[0].shape[1]
        #label_data_resize = np.empty(shape = [len(label_batches), max_num_label, labels_dims])

        ## benz_m, in the new version, it seems that you do not have to make consistent dimension
        # for ii in range(len(label_batches)):
        #    label_data_resize[ii,:,:] = np.resize(label_batches[ii], [max_num_label , labels_dims])

        # sample_weights = np.concatenate(sample_weights,axis=0)
        sg_bidxmaps = np.concatenate(sg_bidxmaps_ls,0)
        # flatten_bidxmaps = np.concatenate(flatten_bidxmaps_ls,0)
        # fmap_neighbor_idises = np.concatenate( fmap_neighbor_idis_ls,0 )
        globalb_bottom_center_xyz = np.concatenate(globalb_bottom_center_xyz_ls, 0)
        fid_start_end = np.concatenate(fid_start_end_ls,0)
        # xyz_mid_batches = np.concatenate( xyz_mid_ls,0 )
        # return data_batches,label_batches,sample_weights,sg_bidxmaps,flatten_bidxmaps, fmap_neighbor_idises,fid_start_end, xyz_mid_batches
        if IsRecordTime:
            t3 = time.time()
            print('t1:%f\nt2:%f\nt3:%f'%(t1-t0,t2-t1,time.time()-t2))
        return data_batches, label_batches, sg_bidxmaps,  globalb_bottom_center_xyz, fid_start_end

    def update_train_eval_shuffled_idx(self):
        flag = 'shuffle_within_each_file'
        if flag == 'shuffle_within_each_file':
            if self.train_file_N>0:
                train_shuffled_idxs = []
                for k in range(self.train_file_N):
                    train_shuffled_idx_k = np.arange( self.g_block_idxs[k,0], self.g_block_idxs[k,1] )
                    np.random.shuffle(train_shuffled_idx_k)
                    train_shuffled_idxs.append( train_shuffled_idx_k )
                self.train_shuffled_idx = np.concatenate( train_shuffled_idxs )

            if self.eval_file_N>0:
                eval_shuffled_idxs = []
                for k in range(self.eval_file_N):
                    eval_shuffled_idx_k = np.arange( self.g_block_idxs[k+self.train_file_N,0], self.g_block_idxs[k+self.train_file_N,1] ) - self.train_num_blocks
                    np.random.shuffle(eval_shuffled_idx_k)
                    eval_shuffled_idxs.append( eval_shuffled_idx_k )
                self.eval_shuffled_idx = np.concatenate( eval_shuffled_idxs )

        if flag == 'shuffle_within_all':
            self.train_shuffled_idx = np.arange(self.train_num_blocks)
            np.random.shuffle(self.train_shuffled_idx)
            self.eval_shuffled_idx = np.arange(self.eval_num_blocks)
            np.random.shuffle(self.eval_shuffled_idx)

    def get_train_batch(self,train_start_batch_idx,train_end_batch_idx,IsShuffleIdx=True):
        assert(train_start_batch_idx>=0 and train_start_batch_idx<=self.train_num_blocks)
        assert(train_end_batch_idx>=0 and train_end_batch_idx<=self.train_num_blocks)
        # all train files are before eval files
        if IsShuffleIdx:
            g_shuffled_batch_idx = self.train_shuffled_idx[range(train_start_batch_idx,train_end_batch_idx)]
            return self.get_shuffled_global_batch(g_shuffled_batch_idx)
        else:
            #return self.get_global_batch(train_start_batch_idx,train_end_batch_idx)
            g_eval_idx = range(train_start_batch_idx, train_end_batch_idx)
            return self.get_shuffled_global_batch(g_eval_idx)


    def get_eval_batch(self,eval_start_batch_idx,eval_end_batch_idx,IsShuffleIdx=False):
        assert(eval_start_batch_idx>=0 and eval_start_batch_idx<=self.eval_num_blocks),"eval_start_batch_idx = %d,  eval_num_blocks=%d"%(eval_start_batch_idx,self.eval_num_blocks)
        assert(eval_end_batch_idx>=0 and eval_end_batch_idx<=self.eval_num_blocks),"eval_end_batch_idx = %d,  eval_num_blocks=%d"%(eval_end_batch_idx,self.eval_num_blocks)

        if IsShuffleIdx:
            g_shuffled_batch_idx = self.eval_shuffled_idx[range(eval_start_batch_idx,eval_end_batch_idx)] + self.eval_global_start_idx
            return self.get_shuffled_global_batch(g_shuffled_batch_idx)
        else:
            eval_start_batch_idx  += self.eval_global_start_idx
            eval_end_batch_idx  += self.eval_global_start_idx
            g_eval_idx = range(eval_start_batch_idx, eval_end_batch_idx)
            return self.get_shuffled_global_batch(g_eval_idx)
            #return self.get_global_batch(eval_start_batch_idx,eval_end_batch_idx)



    def gen_gt_pred_objs(self,visu_fn_glob='The glob for file to be visualized',obj_dump_dir=None):
        for k,norm_h5f in enumerate(self.norm_h5f_L):
            if norm_h5f.file_name.find(visu_fn_glob) > 0:
                norm_h5f.gen_gt_pred_obj( obj_dump_dir )

    def sample(self,data_batches,label_batches,num_point_block):
        NUM_POINT_IN = data_batches.shape[1]
        if num_point_block == None:
            num_point_block = NUM_POINT_IN
        if NUM_POINT_IN != num_point_block:
            sample_choice = GLOBAL_PARA.sample(NUM_POINT_IN,num_point_block,'random')
            data_batches = data_batches[:,sample_choice,...]
            label_batches = label_batches[:,sample_choice]
        return data_batches,label_batches

    def update_sample_loss_weight(self):
        # amount is larger, the loss weight is smaller
        # get all the labels
        train_labels_hist_1norm = []
        test_labels_hist_1norm = []
        labels_hist_1norm = []
        labels_weights = []

        label_name = 'label_category'
        #for label_name in self.norm_h5f_L[0].label_set_elements:
        if label_name in self.feed_label_elements:
            label_hist = np.zeros(self.num_classes).astype(np.int64)
            train_label_hist = np.zeros(self.num_classes).astype(np.int64)
            test_label_hist = np.zeros(self.num_classes).astype(np.int64)
            for k,norme_h5f in enumerate(self.norm_h5f_L):
                label_hist_k = norme_h5f.labels_set.attrs[label_name+'_hist']
                label_hist += label_hist_k.astype(np.int64)
                if k < self.train_file_N:
                    train_label_hist += label_hist_k.astype(np.int64)
                else:
                    test_label_hist += label_hist_k.astype(np.int64)
            train_labels_hist_1norm.append( np.expand_dims( train_label_hist / np.sum(train_label_hist).astype(float),axis=-1) )
            test_labels_hist_1norm.append(   np.expand_dims(test_label_hist / np.sum(test_label_hist).astype(float),axis=-1) )
            cur_labels_hist_1norm =  np.expand_dims(label_hist / np.sum(label_hist).astype(float),axis=-1)
            labels_hist_1norm.append(  cur_labels_hist_1norm )
            labels_weights.append(  1/np.log(1.015+cur_labels_hist_1norm) )
        self.train_labels_hist_1norm = np.concatenate( train_labels_hist_1norm,axis=-1 )
        self.test_labels_hist_1norm = np.concatenate( test_labels_hist_1norm,axis=-1 )
        self.labels_hist_1norm = np.concatenate( labels_hist_1norm,axis=-1 )
        self.labels_weights = np.concatenate( labels_weights,axis=1 )
        self.labels_weights /= self.labels_weights.min()


    def write_file_accuracies(self,obj_dump_dir=None):
        Write_all_file_accuracies(self.sph5_file_list,obj_dump_dir)


def main_NormedH5f():
    '''
    error global blocks in 17DRP5sb8fy region1:  1, 9,11,12,15,16
    '''
    t0 = time.time()
    dataset_name = 'matterport3d'
    dataset_name = 'Voxel'

    all_fn_globs = ['Merged_sph5/32768_gs-100_-100/']
    bxmh5_folder_name = 'Merged_bxmh5/32768_gs-100_-100_fmn-1-12800_6400_560-16_8_8-0d4_0d8_1d8-0d2_0d4_0d4-pd1-3D3'
    eval_fnglob_or_rate = '0.2'

    #all_fn_globs = ['Org_sph5/9000_gs-4_-6d3/']
    #bxmh5_folder_name = 'Org_bxmh5/9000_gs-4_-6d3_fmn1-320_32-320_48-0d9_2d7-0d6_1d8-pd3-TMP'
    #eval_fnglob_or_rate = 'scene0000_00'

    #eval_fnglob_or_rate = 'region0'

    #all_filename_glob = ['all_merged_nf5']
    #eval_fnglob_or_rate = '17DRP5sb8fy'
    num_point_block = None

    only_evaluate = False
    Feed_Data_Elements = ['xyz','xyz_1norm_file','xyz_midnorm_block']
    Feed_Data_Elements = ['xyz','xyz_midnorm_block', 'color_1norm']
    Feed_Data_Elements = ['xyz']
    # Feed_Label_Elements = ['label_category','label_instance']
    Feed_Label_Elements = ['bounding_box']
    net_configs = {}
    net_configs['loss_weight'] = 'E'
    net_provider = Net_Provider_kitti(
                                net_configs=net_configs,
                                dataset_name=dataset_name,
                                all_filename_glob=all_fn_globs,
                                eval_fnglob_or_rate=eval_fnglob_or_rate,
                                bxmh5_folder_name = bxmh5_folder_name,
                                only_evaluate = False,
                                feed_data_elements=Feed_Data_Elements,
                                feed_label_elements=Feed_Label_Elements)
    t1 = time.time()
    print(net_provider.data_summary_str)
    print('init time:',t1-t0)


    #ply_flag = 'region'
    ply_flag = 'global_block'
    #ply_flag = 'sub_block'
    #ply_flag = 'none'
    steps = { 'region':net_provider.eval_num_blocks, 'global_block':1, 'sub_block':1,'none':8 }
    IsShuffleIdx = True

    # cur_data,cur_label,cur_smp_weights,cur_sg_bidxmaps,cur_flatten_bidxmaps, cur_fmap_neighbor_idis, fid_start_end, cur_xyz_mid = net_provider.get_train_batch(2,7,IsShuffleIdx)
    cur_data, cur_label, cur_sg_bidxmaps, fid_start_end = net_provider.get_train_batch(2,7,IsShuffleIdx)


    for bk in  range(0,net_provider.eval_num_blocks,steps[ply_flag]):
        #end = min(bk+s,net_provider.eval_num_blocks)
        #end = net_provider.eval_num_blocks
        end = min( bk+steps[ply_flag], net_provider.eval_num_blocks )
        t0 = time.time()
        cur_data,cur_label,cur_smp_weights,cur_sg_bidxmaps,cur_flatten_bidxmaps, \
            cur_fmap_neighbor_idis, fid_start_end, cur_xyz_mid  = net_provider.get_eval_batch(bk, end, False)
        print('read each block t=%f ms IsShuffleIdx=%s'%( (time.time()-t0)/(end-bk)*1000, IsShuffleIdx ) )
        xyz_idxs = net_provider.feed_data_ele_idxs['xyz']
        color_idxs = net_provider.feed_data_ele_idxs['color_1norm']
        xyz = cur_data[...,xyz_idxs]
        color0 = cur_data[...,color_idxs] * 255
        color = color0.astype(np.uint8)
        xyz_color = np.concatenate( [xyz,color],axis=-1 )

        if ply_flag == 'region':
            ply_fn = DATASET_DIR[dataset_name] + '/PlyFile_17DRP5sb8fy/' + str(eval_fnglob_or_rate)+ '.ply'
            create_ply( xyz_color,ply_fn )
            break
        if ply_flag == 'global_block':
            ply_fn = DATASET_DIR[dataset_name] + '/PlyFile_17DRP5sb8fy/globalblocks/'+ str(eval_fnglob_or_rate) +'/global_b' + str(bk) + '.ply'
            for i in range(xyz_color.shape[0]):
                for j in range(xyz_color.shape[1]):
                    if cur_smp_weights[i,j,0 ] == 0:
                        xyz_color[i,j,3:6] = 0
            create_ply(xyz_color,ply_fn)
            if bk>5:
                break
        if ply_flag == 'sub_block':
            assert xyz_color.ndim == 4
            for sub_k in range(0,xyz_color.shape[1]):
                ply_fn = DATASET_DIR[dataset_name] + '/PlyFile_17DRP5sb8fy/'+ str(eval_fnglob_or_rate) +'/global0_subblocks/sub' + str(sub_k) + '.ply'
                create_ply(xyz_color[0,sub_k,...],ply_fn)
            break

    print(cur_label.shape)
    # print(cur_smp_weights.shape)
    print(cur_data.shape)
    print(cur_data[0,0:3,:])
    print(cur_label[0,0:3,:])
    # print(cur_smp_weights[0,0:3,:])


def check_bxmap_pl_shape_match():
    bxmap_path = '/home/z/Research/dynamic_pointnet/data/Matterport3D_H5F/v1/scans/stride_0d1_step_0d1_bmap_sph5_12800_1d6_2_fmn6-2048_256_64-48_32_16-0d2_0d6_1d2-0d1_0d3_0d6/2n8kARJN3HM'
    pl_path = '/home/z/Research/dynamic_pointnet/data/Matterport3D_H5F/v1/scans/stride_0d1_step_0d1_pl_sph5_1d6_2/2n8kARJN3HM'
    pl_fn_ls = glob.glob( pl_path + '/*.sph5' )
    for pl_fn in pl_fn_ls:
        base_fn = os.path.splitext( os.path.basename( pl_fn ) )[0] + '.bxmh5'
        bxmap_fn = bxmap_path + '/' + base_fn
        if not os.path.exists(bxmap_fn):
            print('%s not exist'%(bxmap_fn))
            import pdb; pdb.set_trace()  # XXX BREAKPOINT
        with h5py.File( bxmap_fn, 'r' ) as bxmf:
          with h5py.File( pl_fn, 'r' ) as plf:
            if bxmf['bidxmaps_flat'].shape[0] != plf['data'].shape[0]:
                print(pl_fn)
                print('shape mathch err')

if __name__=='__main__':
    main_NormedH5f()
    #check_bxmap_pl_shape_match()