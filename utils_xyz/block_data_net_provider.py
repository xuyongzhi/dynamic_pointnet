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


ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR,'data')
DATASET_DIR={}
DATASET_DIR['scannet'] = os.path.join(DATA_DIR,'scannet_data')
DATASET_DIR['stanford_indoor3d'] = os.path.join(DATA_DIR,'stanford_indoor3d')
matterport3D_h5f_dir = os.path.join(DATA_DIR,'Matterport3D_H5F')
DATASET_DIR['matterport3d'] = matterport3D_h5f_dir

#-------------------------------------------------------------------------------
# provider for training and testing
#------------------------------------------------------------------------------
class Net_Provider():
    '''
    (1) provide data for training
    (2) load file list to list of Norm_H5f[]
    dataset_name: 'stanford_indoor3d' 'scannet'
    all_filename_glob:  stride_1_step_2_test_small_4096_normed/*.nh5
    eval_fnglob_or_rate: file name str glob or file number rate. 'scan1*.nh5' 0.2
    num_point_block: if the block point number is not this, do randomly sample
    feed_data_elements: sub list of ['xyz_1norm','xyz_midnorm','nxnynz','color_1norm','intensity_1norm']
    feed_label_elements: sub list of ['label_category','label_instance','label_material']
    '''
    # input normalized h5f files
    # normed_h5f['data']: [blocks*block_num_point*num_channel],like [1000*4096*9]
    # one batch would contain sevel(batch_size) blocks,this will be set out side
    # provider with train_start_idx and test_start_idx

    global_num_point = GlobalSubBaseBLOCK.global_num_point

    def __init__(self,InputType, dataset_name,all_filename_glob,eval_fnglob_or_rate,\
                 only_evaluate,num_point_block=None,feed_data_elements=['xyz_midnorm'],feed_label_elements=['label_category'],\
                 train_num_block_rate=1,eval_num_block_rate=1 ):
        t_init0 = time.time()
        #self.InputType =
        #self.InputType =
        self.InputType = InputType  # 'Sorted_H5f','Normed_H5f','Pr_Normed_H5f'

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
        self.train_file_N = train_file_N = len(train_file_list)
        eval_file_N = len(eval_file_list)
        self.g_file_N = train_file_N + eval_file_N
        self.normed_h5f_file_list =  normed_h5f_file_list = train_file_list + eval_file_list
        #-----------------------------------------------------------------------
        # open each file as a Normed_H5f class instance
        self.norm_h5f_L = []
        # self.g_block_idxs: within the whole train/test dataset  (several files)
        #     record the start/end row idx  of each file to help search data from all files
        #     [ [start_global_row_idxs,end_global__idxs] ]
        #     [[  0,  38], [ 38,  90],[ 90, 150],...[259, 303],[303, 361],[361, 387]]
        #   self.train_num_blocks: 303
        #   self.eval_num_blocks: 84
        #   self.eval_global_start_idx: 303
        self.g_block_idxs = np.zeros((self.g_file_N,2),np.int32)
        self.eval_global_start_idx = None

        file_N = len(normed_h5f_file_list)
        print('Totally %d files. \nReading the block number in each file.'%(file_N))
        t_start_global_blockid = time.time()
        for i,fn in enumerate(normed_h5f_file_list):
            assert(os.path.exists(fn))
            h5f = h5py.File(fn,open_type)
            if self.InputType=='Normed_H5f' or self.InputType=='Pr_Normed_H5f':
                norm_h5f = Normed_H5f(h5f,fn)
            elif self.InputType=='Sorted_H5f':
                norm_h5f = Sorted_H5f(h5f,fn)
            self.norm_h5f_L.append( norm_h5f )
            file_block_N = self.get_block_n(norm_h5f)
            self.g_block_idxs[i,1] = self.g_block_idxs[i,0] + file_block_N
            if i<self.g_file_N-1:
                self.g_block_idxs[i+1,0] = self.g_block_idxs[i,1]
        t_end_global_blockid = time.time()
        print('global block id reading finished, t=%f ms'%(1000.0*(t_end_global_blockid-t_start_global_blockid)))

        self.eval_global_start_idx = self.g_block_idxs[train_file_N,0]
        if train_file_N > 0:
            self.train_num_blocks = self.g_block_idxs[train_file_N-1,1] # = self.eval_global_start_idx
        else: self.train_num_blocks = 0
        self.eval_num_blocks = self.g_block_idxs[-1,1] - self.train_num_blocks
        self.num_classes = self.norm_h5f_L[0].num_classes
        self.label_ele_idxs_inh5f = self.norm_h5f_L[0].label_ele_idxs
        self.label_eles_inh5f = self.norm_h5f_L[0].label_set_elements

        self.update_sample_loss_weight()
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

        self.get_data_label_shape_info()
        t_get_data_laebl_shape = time.time()
        print('get_data_label_shape t: %f ms'%(1000*(t_get_data_laebl_shape - t_end_update_loss_weight)))
        self.update_data_summary()
        print('Net_Provider init t: %f ms\n\n'%(1000*(time.time()-t_init0)))

    def get_data_label_shape_info(self):
        self.feed_data_ele_idxs,self.feed_label_ele_idxs = self.norm_h5f_L[0].get_feed_ele_ids(self.feed_data_elements,self.feed_label_elements)
        self.data_num_eles = len([idx for e in self.feed_data_ele_idxs for idx in self.feed_data_ele_idxs[e] ])
        self.label_num_eles = len([self.feed_label_ele_idxs[e] for e in self.feed_label_ele_idxs])
        if self.InputType=='Sorted_H5f' or self.InputType=='Pr_Normed_H5f':
            self.cascade_num = GlobalSubBaseBLOCK.cascade_num
            self.whole_train_data_shape = np.array([ self.train_num_blocks,GlobalSubBaseBLOCK.nsubblock_candis[0],
                                                    GlobalSubBaseBLOCK.npoint_subblock_candis[0],self.data_num_eles])
            self.whole_eval_data_shape = np.copy(self.whole_train_data_shape)
            self.whole_eval_data_shape[0] = self.eval_num_blocks
            self.whole_train_label_shape = np.copy(self.whole_train_data_shape)
            self.whole_train_label_shape[-1] = self.label_num_eles

            #self.get_data_label_shape_byread()
        elif self.InputType=='Normed_H5f':
            self.get_data_label_shape_byread()
        block_sample = self.whole_train_data_shape[1:-1]
        if block_sample.size==1:
            block_sample= block_sample[0]
        elif block_sample.size==2:
            block_sample = (block_sample[0],block_sample[1])
        self.block_sample = block_sample

        if self.InputType == 'Pr_Normed_H5f':
            self.sg_bidxmaps_shape = GlobalSubBaseBLOCK.get_sg_bidxmaps_fixed_shape()
            self.flatten_bidxmaps_shape = GlobalSubBaseBLOCK.get_flatten_bidxmaps_shape()
            self.sg_bidxmaps_extract_idx = GlobalSubBaseBLOCK.sg_bidxmaps_extract_idx
            self.flatten_bidxmaps_extract_idx = GlobalSubBaseBLOCK.flatten_bidxmaps_extract_idx

        print('\ntrain data shape',self.whole_train_data_shape)
        print('train label shape',self.whole_train_label_shape)
        print('eval data shape',self.whole_eval_data_shape)

    def get_data_label_shape_byread(self):
        t0 = time.time()
        data_batches,label_batches,sample_weights,sg_bidxmaps,flatten_bidxmaps = self.get_train_batch(0,min(self.train_num_blocks,32))
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
        if self.InputType != 'Pr_Normed_H5f':
            return True
         datas,labels,sample_weights,sg_bidxmaps,flatten_bidxmaps = self.get_train_batch(0,min(self.train_num_blocks,32))



    def get_block_n(self,norm_h5f):
        if self.InputType == 'Normed_H5f' or self.InputType=='Pr_Normed_H5f':
            file_block_N = norm_h5f.data_set.shape[0]
        elif self.InputType == 'Sorted_H5f':
            file_block_N = GlobalSubBaseBLOCK.get_block_n_of_new_stride_step_(norm_h5f.h5f,norm_h5f.file_name,'global')
        return file_block_N

    def update_data_summary(self):
        self.data_summary_str = '%s \nfeed_data_elements:%s \nfeed_label_elements:%s \n'%(self.dataset_name,self.feed_data_elements,self.feed_label_elements)
        self.data_summary_str += 'train data shape: %s \ntest data shape: %s \n'%(
                         str(self.whole_train_data_shape),str(self.whole_eval_data_shape))
       # self.data_summary_str += 'train labels histogram: %s \n'%( np.array_str(np.transpose(self.train_labels_hist_1norm) ))
       # self.data_summary_str += 'test labels histogram: %s \n'%( np.array_str(np.transpose(self.test_labels_hist_1norm) ))
        self.data_summary_str += 'labels histogram: %s \n'%( np.array_str(np.transpose(self.labels_hist_1norm[:,0]) ))
        #print(self.data_summary_str)

    def get_all_file_name_list(self,dataset_name,all_filename_globs):
        all_file_list = []
        fn_globs = []
        if self.InputType=='Normed_H5f':
            file_format = '.nh5'
        elif self.InputType=='Pr_Normed_H5f':
            file_format = '.prh5'
        elif self.InputType=='Sorted_H5f':
            file_format = '.sh5'
        else:
            assert False,"file format err: no "+self.InputType

        for all_filename_glob in all_filename_globs:
            fn_glob = os.path.join(DATASET_DIR[dataset_name],all_filename_glob+'/*'+file_format)
            all_file_list  += glob.glob( fn_glob )
            fn_globs.append(fn_glob)
        if len(all_file_list)== 0:
            print('no file in:')
            print(fn_globs)
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
            # split by number
            n = len(all_file_list)
            m = int(n*(1-eval_fnglob_or_rate))
            train_file_list = all_file_list[0:m]
            eval_file_list = all_file_list[m:n]

        log_str = '\ntrain file list (n=%d) = \n%s\n\n'%(len(train_file_list),train_file_list[-2:])
        log_str += 'eval file list (n=%d) = \n%s\n\n'%(len(eval_file_list),eval_file_list[-2:])
        print( log_str )
        return train_file_list,eval_file_list


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
        assert self.InputType=='Normed_H5f' or self.InputType=='Pr_Normed_H5f'
        if self.InputType=='Normed_H5f' or self.InputType=='Pr_Normed_H5f':
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



    def get_global_batch(self,g_start_idx,g_end_idx):
        start_file_idx,end_file_idx,local_start_idx,local_end_idx = \
            self.global_idx_to_local(g_start_idx,g_end_idx)
        #t0 = time.time()

        data_ls = []
        label_ls = []
        center_mask = []
        sg_bidxmaps_ls = []
        flatten_bidxmaps_ls = []
        for f_idx in range(start_file_idx,end_file_idx+1):
            if f_idx == start_file_idx:
                start = local_start_idx
            else:
                start = 0
            if f_idx == end_file_idx:
                end = local_end_idx
            else:
                end = self.get_block_n(self.norm_h5f_L[f_idx])

            IsInclude_xyz_midnorm_block = 'xyz_midnorm_block' in self.feed_data_elements
            feed_data_elements = self.feed_data_elements
            if not IsInclude_xyz_midnorm_block:
                new_feed_data_elements = feed_data_elements + ['xyz_midnorm_block']
                new_feed_data_ele_idxs,_ = self.norm_h5f_L[0].get_feed_ele_ids(new_feed_data_elements,self.feed_label_elements)
            else:
                new_feed_data_elements = feed_data_elements
            if self.InputType == 'Normed_H5f' or self.InputType=='Pr_Normed_H5f':
                data_i = self.norm_h5f_L[f_idx].get_normed_data(start,end,new_feed_data_elements)
                label_i = self.norm_h5f_L[f_idx].get_label_eles(start,end,self.feed_label_elements)
                # data_i: [batch_size,npoint_block,data_nchannels]
                # label_i: [batch_size,npoint_block,label_nchannels]
                if self.InputType=='Pr_Normed_H5f':
                    sg_bidxmaps, flatten_bidxmaps = self.norm_h5f_L[f_idx].get_bidxmap(start,end)
            elif self.InputType == 'Sorted_H5f':
                data_i,label_i = self.norm_h5f_L[f_idx].get_batch_of_larger_block(
                                start,end,new_feed_data_elements,self.feed_label_elements )
                # data_i: [batch_size,nsubblock,npoint_subblock,data_nchannels]
                # label_i: [batch_size,nsubblock,npoint_subblock,label_nchannels]
            assert data_i.ndim == label_i.ndim and (data_i.shape[0:-1] == label_i.shape[0:-1])
            if not IsInclude_xyz_midnorm_block:
                data_i = np.delete(data_i,new_feed_data_ele_idxs['xyz_midnorm_block'],axis=-1)

            data_ls.append(data_i)
            label_ls.append(label_i)
            if self.InputType=='Pr_Normed_H5f':
                sg_bidxmaps_ls.append(sg_bidxmaps)
                flatten_bidxmaps_ls.append(flatten_bidxmaps)

            xyz_midnorm_block_i = data_i[...,self.feed_data_ele_idxs['xyz_midnorm_block']]

            center_mask_i = self.get_center_mask(xyz_midnorm_block_i)
            center_mask.append(center_mask_i)

        data_batches = np.concatenate(data_ls,0)
        label_batches = np.concatenate(label_ls,0)
        if self.InputType=='Pr_Normed_H5f':
            sg_bidxmaps = np.concatenate( sg_bidxmaps_ls,axis=0 )
            flatten_bidxmaps = np.concatenate( flatten_bidxmaps_ls,axis=0 )
        else:
            sg_bidxmaps = None
            flatten_bidxmaps = None

        center_mask = np.concatenate(center_mask,0)

        #if self.InputType=='Normed_H5f' or self.InputType=='Pr_Normed_H5f':
            # sampling again
            #if self.num_point_block!=None:
                #assert data_batches.shape[1] == self.num_point_block
                #data_batches,label_batches = self.sample(data_batches,label_batches,self.num_point_block)

        num_label_eles = len(self.feed_label_elements)
        center_mask = np.expand_dims(center_mask,axis=-1)
        center_mask = np.tile(center_mask,(1,1,num_label_eles))
        sample_weights = []
        for k in range(num_label_eles):
            if k == self.feed_label_ele_idxs['label_category'][0]:
                sample_weights_k = np.take(self.labels_weights[:,k],label_batches[...,k])
            else:
                sample_weights_k = np.ones_like(label_batches[...,k])
            sample_weights.append( np.expand_dims(sample_weights_k,axis=-1) )
        sample_weights = np.concatenate(sample_weights,axis=-1)

        sample_weights *= center_mask

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

        return data_batches,label_batches,sample_weights,sg_bidxmaps,flatten_bidxmaps

    def get_center_mask(self,xyz_midnorm,edge_rate=0.12):
        # true for center, false for edge
        # edge_rate: distance to center of the block_step
        block_step = self.norm_h5f_L[0].h5f.attrs['block_step']
        center_rate = np.abs(xyz_midnorm / block_step) # -0.5 ~ 0.5
        center_mask = (center_rate[...,0] < (0.5-edge_rate)) * ( center_rate[...,0] < (0.5-edge_rate) )
        #print('center n rate= %f'%(np.sum(center_mask).astype(float)/xyz_midnorm.shape[0]/xyz_midnorm.shape[1]))
        return center_mask

    def get_shuffled_global_batch(self,g_shuffled_idx_ls):
        data_batches = []
        label_batches = []
        sample_weights = []
        sg_bidxmaps_ls = []
        flatten_bidxmaps_ls = []
        for idx in g_shuffled_idx_ls:
            data_i,label_i,smw_i,sg_bidxmaps_i,flatten_bidxmaps_i = self.get_global_batch(idx,idx+1)
            if self.InputType=='Pr_Normed_H5f':
                sg_bidxmaps_ls.append(sg_bidxmaps_i)
                flatten_bidxmaps_ls.append(flatten_bidxmaps_i)
            data_batches.append(data_i)
            label_batches.append(label_i)
            sample_weights.append(smw_i)
        data_batches = np.concatenate(data_batches,axis=0)
        label_batches = np.concatenate(label_batches,axis=0)
        sample_weights = np.concatenate(sample_weights,axis=0)
        if self.InputType=='Pr_Normed_H5f':
            sg_bidxmaps = np.concatenate(sg_bidxmaps_ls,0)
            flatten_bidxmaps = np.concatenate(flatten_bidxmaps_ls,0)
        else:
            sg_bidxmaps = None
            flatten_bidxmaps = None
        return data_batches,label_batches,sample_weights,sg_bidxmaps,flatten_bidxmaps

    def update_train_eval_shuffled_idx(self):
        self.train_shuffled_idx = np.arange(self.train_num_blocks)
        np.random.shuffle(self.train_shuffled_idx)
        self.eval_shuffled_idx = np.arange(self.eval_num_blocks)
        np.random.shuffle(self.eval_shuffled_idx)

    def get_train_batch(self,train_start_batch_idx,train_end_batch_idx):
        assert(train_start_batch_idx>=0 and train_start_batch_idx<=self.train_num_blocks)
        assert(train_end_batch_idx>=0 and train_end_batch_idx<=self.train_num_blocks)
        # all train files are before eval files
        IsShuffleIdx = False
        if IsShuffleIdx:
            g_shuffled_batch_idx = self.train_shuffled_idx[range(train_start_batch_idx,train_end_batch_idx)]
            return self.get_shuffled_global_batch(g_shuffled_batch_idx)
        else:
            return self.get_global_batch(train_start_batch_idx,train_end_batch_idx)

    def get_eval_batch(self,eval_start_batch_idx,eval_end_batch_idx):
        assert(eval_start_batch_idx>=0 and eval_start_batch_idx<=self.eval_num_blocks)
        assert(eval_end_batch_idx>=0 and eval_end_batch_idx<=self.eval_num_blocks)

        IsShuffleIdx = True
        if IsShuffleIdx:
            g_shuffled_batch_idx = self.eval_shuffled_idx[range(eval_start_batch_idx,eval_end_batch_idx)] + self.eval_global_start_idx
            return self.get_shuffled_global_batch(g_shuffled_batch_idx)
        else:
            eval_start_batch_idx  += self.eval_global_start_idx
            eval_end_batch_idx  += self.eval_global_start_idx
            return self.get_global_batch(eval_start_batch_idx,eval_end_batch_idx)


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
                if self.InputType == 'Normed_H5f' or self.InputType=='Pr_Normed_H5f':
                    label_hist_k = norme_h5f.labels_set.attrs[label_name+'_hist']
                elif self.InputType == 'Sorted_H5f':
                    label_hist_k = norme_h5f.h5f.attrs[label_name+'_hist']
                label_hist += label_hist_k.astype(np.int64)
                if k < self.train_file_N:
                    train_label_hist += label_hist_k.astype(np.int64)
                else:
                    test_label_hist += label_hist_k.astype(np.int64)
            train_labels_hist_1norm.append( np.expand_dims( train_label_hist / np.sum(train_label_hist).astype(float),axis=-1) )
            test_labels_hist_1norm.append(   np.expand_dims(test_label_hist / np.sum(test_label_hist).astype(float),axis=-1) )
            cur_labels_hist_1norm =  np.expand_dims(label_hist / np.sum(label_hist).astype(float),axis=-1)
            labels_hist_1norm.append(  cur_labels_hist_1norm )
            labels_weights.append(  1/np.log(1.2+cur_labels_hist_1norm) )
        self.train_labels_hist_1norm = np.concatenate( train_labels_hist_1norm,axis=-1 )
        self.test_labels_hist_1norm = np.concatenate( test_labels_hist_1norm,axis=-1 )
        self.labels_hist_1norm = np.concatenate( labels_hist_1norm,axis=-1 )
        self.labels_weights = np.concatenate( labels_weights,axis=1 )


    def write_file_accuracies(self,obj_dump_dir=None):
        Write_all_file_accuracies(self.normed_h5f_file_list,obj_dump_dir)


def main_NormedH5f():
    t0 = time.time()
    dataset_name = 'matterport3d'

    InputType = 'Normed_H5f'
    all_filename_glob = ['all_merged_nf5/v1_scans']
    #num_point_block = 8192

    InputType = 'Pr_Normed_H5f'
    all_filename_glob = ['v1/scans/17DRP5sb8fy/stride_0d1_step_0d1_pyramid-1_2-512_256_64_32-0d2_0d4_0d8_16']
    num_point_block = None

    eval_fnglob_or_rate = 0.3
    only_evaluate = False
    feed_data_elements = ['xyz_1norm_file','xyz_midnorm_block']
    feed_label_elements = ['label_category','label_instance']
    #feed_label_elements = ['label_category','label_instance']
    net_provider=Net_Provider(InputType=InputType,
                              dataset_name=dataset_name,
                              all_filename_glob=all_filename_glob,
                              eval_fnglob_or_rate=eval_fnglob_or_rate,
                              only_evaluate=only_evaluate,
                              num_point_block=num_point_block,
                              feed_data_elements=feed_data_elements,
                              feed_label_elements=feed_label_elements)
    t1 = time.time()
    print(net_provider.data_summary_str)
    print('init time:',t1-t0)

    cur_data,cur_label,cur_smp_weights,cur_sg_bidxmaps,cur_flatten_bidxmaps = net_provider.get_train_batch(0, min(128,net_provider.train_num_blocks) )
    cur_data,cur_label,cur_smp_weights,cur_sg_bidxmaps,cur_flatten_bidxmaps = net_provider.get_eval_batch(0,min(128,net_provider.eval_num_blocks) )
    if InputType=='Normed_H5f':
        print(cur_data.shape)
        print(cur_data[0,0:3,:])
        print(cur_label.shape)
        print(cur_label[0,0:3,:])
        print(cur_smp_weights.shape)
        print(cur_smp_weights[0,0:3,:])
    else:
        print(cur_data.shape)
        print(cur_data[0,0,0:3,:])
        print(cur_label.shape)
        print(cur_label[0,0,0:3,:])
        print(cur_smp_weights.shape)
        print(cur_smp_weights[0,0,0:3,:])


def main_SortedH5f():
    t0 = time.time()
    InputType = 'Sorted_H5f'
    dataset_name = 'matterport3d'
    all_filename_glob = ['v1/scans/17DRP5sb8fy/stride_0d1_step_0d1']
    eval_fnglob_or_rate = 0.3
    only_evaluate = False
    num_point_block = None
    feed_data_elements = ['xyz_1norm_file','xyz_midnorm_block','color_1norm']
    feed_data_elements = ['xyz_midnorm_block']
    feed_label_elements = ['label_category','label_instance']
    #feed_label_elements = ['label_category','label_instance']
    net_provider=Net_Provider(InputType=InputType,
                              dataset_name=dataset_name,
                              all_filename_glob=all_filename_glob,
                              eval_fnglob_or_rate=eval_fnglob_or_rate,
                              only_evaluate=only_evaluate,
                              num_point_block=num_point_block,
                              feed_data_elements=feed_data_elements,
                              feed_label_elements=feed_label_elements)
    t1 = time.time()
    return
    print(net_provider.data_summary_str)
    print('init time:',t1-t0)

    cur_data,cur_label,cur_smp_weights = net_provider.get_train_batch(0, min(128,net_provider.train_num_blocks) )
    cur_data,cur_label,cur_smp_weights = net_provider.get_eval_batch(0,min(128,net_provider.eval_num_blocks) )
    print(cur_data.shape)
    print(cur_data[0,0:3,:])
    print(cur_label.shape)
    print(cur_label[0,0:3,:])
    print(cur_smp_weights.shape)
    print(cur_smp_weights[0,0:3,:])
    print('read 128 blocks t:',time.time()-t1)


if __name__=='__main__':
    main_NormedH5f()
