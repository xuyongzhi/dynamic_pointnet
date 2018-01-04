## Ben, 4 Jan 2018

import numpy as np
import os
import random
import sys
import time
import glob
import h5py


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CFG_DIR  = os.path.join(ROOT_DIR, 'config')
sys.path.append(CFG_DIR)
BENCHMARK = 'KITTI'
KITTI_DATA_DIR = os.path.join(DATA_DIR, BENCHMARK)

from config import cfg

Channels_xyz  = 3
Channels_label =7



class kitti_data_net_provider():
    '''
    (1) collect all the file names in the KITTI benchmark
    (2) doing shuffling
    (3) read a batch for network
    '''
    def __init__(self, rawh5_name, batch_size):
        #self.point_cloud_name = 'velodyne'
        #self.label_name = 'labels'
        #self.rawh5_name = 'rawh5_kitti'
        self.rawh5_name = rawh5_name
        self.batch_size = batch_size

        self.rawh5_file_path = os.path.join(KITTI_DATA_DIR, rawh5_name)
        self.rawh5_file_list = glob.glob(os.path.join(self.rawh5_file_path,'*.h5'))
        self.all_file_name = []
        assert len(self.rawh5_file_list) > 0
        for name in self.rawh5_file_list:
            if name.endswith('.h5'):
                self.all_file_name.append(name.rstrip('.h5'))

        self._shuffle_rawh5_inds()


    def _shuffle_rawh5_inds(self):
        self._perm = np.random.permutation(np.arange(len(self.rawh5_file_list)))
        self._cur  = 0

    def _get_next_minibatch_ind(self):

        if self._cur + self.batch_size >= len(self.rawh5_file_list):
            self._shuffle_rawh5_inds()
        data_inds = self._perm[self._cur:(self._cur+self.batch_size)]
        self._cur = self._cur + self.batch_size
        data_inds.reshape(-1,1)
        return data_inds

    def _get_next_minibatch(self):
        '''
        returen a minibatch including point cloud and labels
        point_cloud_data = [batch, num, 3]
        labels = [batch, num, 7]
        both of them are saved in the format of list, for different point cloud blocks have different number of bounding box
        '''
        ind = self._get_next_minibatch_ind()
        point_cloud_data = []
        label_data = []
        _index_ = 0
        for ii in ind:
            rawh5_file_name = self.rawh5_file_list[ii]
            with h5py.File(rawh5_file_name,'r') as h5f:
                temp_xyz = h5f['xyz'][:,:]
                temp_bounding_box = h5f['bounding_box'][:,:]
                if cfg.TRAIN.USE_FLIPPED and random.choice([True, False]):
                    temp_xyz[:,1] = -temp_xyz[:,1]
                    temp_bounding_box[:,3] = np.pi - temp_bounding_box[:,3]
                    temp_bounding_box[:,5] = - temp_bounding_box[:,5]

                #point_cloud_data.append(h5f['xyz'][:,:])
                point_cloud_data.append(temp_xyz)
                ## adding the flipping function later
                #label_data.append(h5f['bounding_box'][:,:])
                label_data.append(temp_bounding_box)
            _index_ = _index_ + 1

        return point_cloud_data, label_data






if __name__ == '__main__':
    point_cloud_name = 'velodyne'
    label_name = 'labels'
    START_T = time.time()
    rawh5_name = 'rawh5_kitti'
    batch_size = 20
    data_provider = kitti_data_net_provider(rawh5_name, batch_size)

    point_cloud_data, label_data = data_provider._get_next_minibatch() ## both of them are list type

    print(len(point_cloud_data))
    END_T = time.time()
    print('the time is %d' %(END_T - START_T))
