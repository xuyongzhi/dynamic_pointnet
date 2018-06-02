#!/usr/bin/env python

# Xuesong Li 2 Jan 2018

from __future__ import print_function
#import pdb, traceback
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from kitti_block_data_prep_util import Raw_H5f, Sort_RawH5f, show_h5f_summary_info
import numpy as np
import numpy.random as npr
import h5py
import glob
import time
import multiprocessing as mp
import itertools
import zipfile, gzip
import time
from plyfile import PlyData, PlyElement
#from plyfile import plyData, plyElement


TMPDEBUG = False

ROOT_DIR = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(ROOT_DIR,'data')
DATA_SOURCE = 'KITTI'
KITTI_DATA_DIR = os.path.join(DATA_DIR, DATA_SOURCE)

def read_label_from_txt(label_file_name):
    assert os.path.exists(label_file_name)
    bounding_box = []
    with open(label_file_name,'r') as f:
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
    return data[:,0:8]


def read_point_cloud_from_bin(point_cloud_file_name):
    point_cloud= np.fromfile(point_cloud_file_name, dtype = np.float32).reshape(-1, 4)
    # filter kitti point cloud
    return point_cloud
    #inside_index = np.logical_and( (point_cloud[:,1] < point_cloud[:,0] - 0.27), (-point_cloud[:,1] < point_cloud[:,0] - 0.27))
    #return point_cloud[inside_index]

def random_sample_to_fix_num(point_cloud_raw_data , random_point_cloud_num):
    '''
    the step is specific for KITTI benchmark, num of poinc cloud ranges from 19273 to 31670, all the point cloud are randomly sampled to fixed num, random_point_cloud_num
    '''
    length_raw_data = len(point_cloud_raw_data)
    #assert length_raw_data < random_point_cloud_num
    #assert length_raw_data > random_point_cloud_num/2
    if length_raw_data >= random_point_cloud_num:
        select_index = npr.choice( range(length_raw_data), size = random_point_cloud_num, replace = False)
        point_cloud_data = point_cloud_raw_data[select_index,:]
    else:
        padding_index = npr.choice( range(length_raw_data), size=(random_point_cloud_num - length_raw_data), replace = False)
        padding_data  = point_cloud_raw_data[padding_index,:]
        point_cloud_data = np.concatenate((point_cloud_raw_data, padding_data),axis = 0)


    return point_cloud_data



class kitti_prepare():
    '''
    Reading the KITTI data into h5f
    Downsampling the num of point into a certain num
    '''
    benchmark_name = '/KITTI'
    point_cloud_name = 'velodyne'
    label_name  = 'output'
    KITTI_DATA_DIR = '/home/ben/dataset/KITTI/'
    # point_cloud_file_path = os.path.join(KITTI_DATA_DIR,point_cloud_name)
    point_cloud_file_path = '/home/ben/dataset/KITTI/velodyne_raw'
    #label_file_path = os.path.join(KITTI_DATA_DIR, label_name)
    label_file_path = '/home/ben/dataset/KITTI/filtered_label'
    rawh5f_path = os.path.join(KITTI_DATA_DIR,'all_rawh5f')
    sort_raw_path = os.path.join(KITTI_DATA_DIR,'sort_raw_14')
    random_point_cloud_num = 2**14 # 32768


    @staticmethod
    def generate_kitti_to_rawh5f():
        label_file_list = glob.glob(os.path.join(kitti_prepare.label_file_path,'*.txt'))
        print('%d files in \n%s' %(len(label_file_list), kitti_prepare.label_file_path))
        rawh5f_file_path = kitti_prepare.rawh5f_path
        if not os.path.exists(rawh5f_file_path):
            os.makedirs(rawh5f_file_path)
        for l_f_n in label_file_list:
            file_name = os.path.splitext(os.path.basename(l_f_n))[0]
            label_file_name = os.path.join(kitti_prepare.label_file_path, file_name + '.txt')
            point_cloud_file_name = os.path.join(kitti_prepare.point_cloud_file_path, file_name + '.bin')
            h5f_file_name = os.path.join(rawh5f_file_path, file_name +'.rh5')

            with h5py.File(h5f_file_name, 'w') as h5f:
                raw_h5f = Raw_H5f(h5f, h5f_file_name,'KITTI')
                point_cloud_raw_data = read_point_cloud_from_bin(point_cloud_file_name)

                #point_cloud_data = random_sample_to_fix_num(point_cloud_raw_data , kitti_prepare.random_point_cloud_num) ## specifial for kitti dataset
                point_cloud_data = point_cloud_raw_data

                # label_data = read_label_from_txt(label_file_name)

                num_row = point_cloud_data.shape[0]
                raw_h5f.set_num_default_row(num_row)
                raw_h5f.append_to_dset('xyz',point_cloud_data[:,0:3])

                #num_row = label_data.shape[0]
                #raw_h5f.set_num_default_row(num_row)
                #raw_h5f.append_to_dset('bounding_box', label_data)
                raw_h5f.create_done()

    @staticmethod
    def read_rawh5f():
        rawh5f_list = glob.glob(os.path.join(kitti_prepare.rawh5f_path,'*.h5'))
        for file_name in rawh5f_list:
            with h5py.File(file_name,'r') as h5f:
                #raw_h5f = Raw_H5f(h5f,file_name)
                raw_data = h5f['xyz'][:,:]
                label_data = h5f['bounding_box'][:,:]

    @staticmethod
    def sort_raw():
        raw_file_list = glob.glob(os.path.join(kitti_prepare.rawh5f_path, '*.h5'))
        block_step = [0.5,0.5,0.5]
        print('%d files in %s' %(len(raw_file_list), kitti_prepare.rawh5f_path))
        sort_raw_file_path = kitti_prepare.sort_raw_path
        if not os.path.exists(sort_raw_file_path):
            os.makedirs(sort_raw_file_path)
        Sort_RawH5f(raw_file_list, block_step, sort_raw_file_path)



    @staticmethod
    def showfilesummary(file_name):
        h5f_file_name = kitti_prepare.sort_raw_path + file_name
        with h5py.File(h5f_file_name,'r') as h5f:
            show_h5f_summary_info(h5f)


if __name__ == '__main__':
    START_T = time.time()
    benchmark_name = '/KITTI'
    dataset_names  = ['velodnye']
    file_name = '/000059.sh5'

    kitti_prepare.generate_kitti_to_rawh5f()
    # kitti_prepare.read_rawh5f()
    #kitti_prepare.sort_raw()
    #kitti_prepare.showfilesummary(file_name)

    T = time.time() - START_T
