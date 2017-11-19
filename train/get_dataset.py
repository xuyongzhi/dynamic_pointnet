# xyz Nov 2017

import numpy as np
import time
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR,'data')
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,'utils'))
sys.path.append(os.path.join(ROOT_DIR,'models'))
sys.path.append(os.path.join(ROOT_DIR,'scannet'))
import provider
import scannet_dataset
import time
from collections import deque


def get_channel_indexs(channel_elementes):
    channel_indexs_dict = {
        'xyz_midnorm':[0,1,2],
        'color_1norm':[3,4,5],
        'xyz_1norm':  [6,7,8]}
    channel_indexs = []
    for e in channel_elementes:
        channel_indexs += channel_indexs_dict[e]
    channel_indexs.sort()
    num_channels = len(channel_indexs)
    return channel_indexs,num_channels

def Load_Stanford_Sampled_Hdf5(test_area,channel_elementes = ['xyz_1norm'],max_test_fn=None):
    CHANNEL_INDEXS,NUM_CHANNELS = get_channel_indexs(channel_elementes)

    ALL_FILES = provider.getDataFiles('../data/indoor3d_sem_seg_hdf5_data/all_files.txt')
    room_filelist = [line.rstrip() for line in open('../data/indoor3d_sem_seg_hdf5_data/room_filelist.txt')]

    data_batch_list = []
    label_batch_list = []
    for i,h5_filename in enumerate(ALL_FILES):
        data_batch, label_batch = provider.loadDataFile(h5_filename)
        data_batch = data_batch[...,CHANNEL_INDEXS]
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)

    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)

    test_area_str = 'Area_'+str(test_area)

    train_idxs = []
    test_idxs = []
    for i,room_name in enumerate(room_filelist):
        if test_area_str in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)

    if max_test_fn != None and len(test_idxs)>max_test_fn:
        if len(test_idxs)>max_test_fn:
            test_idxs = test_idxs[0:max_test_fn]
        if len(train_idxs)>max_test_fn*5:
            train_idxs = train_idxs[0:max_test_fn*5]
        print('\n!!! in  small data scale: only read %d test files\n'%(max_test_fn))

    train_data = data_batches[train_idxs,...]
    train_label = label_batches[train_idxs]
    test_data = data_batches[test_idxs,...]
    test_label = label_batches[test_idxs]
    return train_data,train_label,test_data,test_label

def Load_Scannet(npoints=8192):
    data_root = os.path.join(DATA_DIR,'scannet_data')
    scannet_data_test = scannet_dataset.ScannetDatasetWholeScene(
        root = data_root,npoints=npoints,split = 'test')
    scannet_data_train = scannet_dataset.ScannetDatasetWholeScene(
        root = data_root,npoints=npoints,split = 'train')
    return scannet_data_train,scannet_data_test

class GetDataset():
    def __init__(self,data_source,num_point=8192,test_area=6,channel_elementes=['xyz_1norm'],
                 max_test_fn=None):
        self.num_blocks = {}
        self.shuffled_idx = {}
        if data_source == 'stanford_indoor':
            self.num_classes = 13
            data = {}
            label = {}
            data['train'],label['train'],data['test'],label['test'] =  Load_Stanford_Sampled_Hdf5(
                test_area,channel_elementes,max_test_fn )
            for tot in ['train','test']:
                self.num_blocks[tot] = data[tot].shape[0]
                self.shuffled_idx[tot] = np.arange(self.num_blocks[tot])
            self.num_channels = data['train'].shape[2]
            self.data = data
            self.label = label
        elif data_source == 'scannet':
            self.num_classes = 22
            scannet_ds = {}
            self.num_scans = {}
            self.scannet_scan_idx = {}
            self.scannet_buf = {}

            t0 = time.time()
            scannet_ds['train'],scannet_ds['test'] = Load_Scannet(num_point)
            t1 = time.time() - t0
            print('load scannet t = ',t1)

            for tot in ['train','test']:
                self.num_blocks[tot] = None # the num of blocks is unknown, because the blocks in each scan is varaint
                self.num_scans[tot] = len(scannet_ds[tot])
                self.shuffled_idx[tot] = np.arange(self.num_scans[tot])
                self.scannet_scan_idx[tot] = 0 # the next scan idx to read in self.scannet_ds
                self.scannet_buf[tot] = ( np.array([]),np.array([]),np.array([]) )
            self.num_channels = scannet_ds['test'][0][0].shape[2]
            self.scannet_ds = scannet_ds
        self.data_source = data_source
        self.num_point = num_point
        self.shuffle_idx()

    def shuffle_idx(self):
        # during training, this should be performed per epoch
        for tot in ['train','test']:
            np.random.shuffle(self.shuffled_idx[tot])


    def train_dlw(self,batch_idx_start=None,batch_idx_end=None):
        return self.get_data_label_weights('train',batch_idx_start,batch_idx_end)
    def test_dlw(self,batch_idx_start=None,batch_idx_end=None):
        return self.get_data_label_weights('test',batch_idx_start,batch_idx_end)

    def get_data_label_weights(self,tot,batch_idx_start=None,batch_idx_end=None):
        if batch_idx_start==None:
            batch_idx_start = 0
            batch_idx_end = self.num_blocks[tot]
        elif batch_idx_end == None:
            batch_idx_end = batch_idx_start+1
        batch_size = batch_idx_end - batch_idx_start

        if self.data_source == 'stanford_indoor':
            shuffled_idxs = self.shuffled_idx[tot][batch_idx_start:batch_idx_end]
            data_cur = self.data[tot][shuffled_idxs,:,:]
            label_cur = self.label[tot][shuffled_idxs,:]
            BATCH_SIZE = data_cur.shape[0]
            sample_weights_cur = np.ones([BATCH_SIZE,self.num_point],dtype=np.float32)
            return data_cur,label_cur,sample_weights_cur
        elif self.data_source == 'scannet':
            if self.scannet_buf[tot][0].shape[0] <batch_size:
                if self.add_scannet_buf(tot)==False:
                    return None,None,None # reading finished
            dlw = []
            for i in range(len(self.scannet_buf[tot])):
                dlw.append( self.scannet_buf[tot][i][0:batch_size,...] )
                np.delete(self.scannet_buf[tot][i],range(0,batch_size),axis=0)

            return dlw[0],dlw[1],dlw[2]


    def add_scannet_buf(self,tot):
        if self.scannet_scan_idx[tot] >= self.num_scans[tot]:
            return False
        shuffled_block_idx =  self.shuffled_idx[tot][self.scannet_scan_idx[tot]]
        dlw = self.scannet_ds[tot][shuffled_block_idx]
        if self.scannet_buf[tot][0].shape[0] != 0:
            for i in range(len(dlw)):
                    self.scannet_buf[tot][i] = np.concatenate([self.scannet_buf[tot][i],dlw[i]],axis=0)
        else:
            self.scannet_buf[tot] = dlw
        self.scannet_scan_idx[tot] += 1
        return True



