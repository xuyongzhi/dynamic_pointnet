#!/usr/bin/env python

# Ben 2 Jan 2018

from __future__ import print_function
#import pdb, traceback
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from block_data_prep_util import Raw_H5f, Sort_RawH5f,Sorted_H5f,Normed_H5f,show_h5f_summary_info,MergeNormed_H5f
#from block_data_prep_util import  Raw_H5f,Sort_RawH5f,Sorted_H5f
import numpy as np
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
                bounding_box.append(label[1:8])
    assert bounding_box
    data = np.array(bounding_box, dtype = np.float32)
    return data[:,0:7]


def read_point_cloud_from_bin(point_cloud_file_name):
    point_cloud= np.fromfile(point_cloud_file_name, dtype = np.float32).reshape(-1, 4)
    # filter kitti point cloud
    inside_index = np.logical_and( (point_cloud[:,1] < point_cloud[:,0] - 0.27), (-point_cloud[:,1] < point_cloud[:,0] - 0.27))
    return point_cloud[inside_index]

class kitti_prepare():
    '''
    Reading the KITTI data into h5f
    Downsampling the num of point into a certain num
    '''
    benchmark_name = '/KITTI'
    point_cloud_name = 'velodyne'
    label_name  = 'labels'
    point_cloud_file_path = os.path.join(KITTI_DATA_DIR,point_cloud_name)
    label_file_path = os.path.join(KITTI_DATA_DIR, label_name)
    rawh5f_path = os.path.join(KITTI_DATA_DIR,'rawh5f')



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
            h5f_file_name = os.path.join(rawh5f_file_path, file_name +'.h5')

            with h5py.File(h5f_file_name, 'w') as h5f:
                raw_h5f = Raw_H5f(h5f, h5f_file_name,'KITTI')
                point_cloud_data = read_point_cloud_from_bin(point_cloud_file_name)
                label_data = read_label_from_txt(label_file_name)
                num_row = point_cloud_data.shape[0]
                raw_h5f.set_num_default_row(num_row)
                raw_h5f.append_to_dset('xyz',point_cloud_data[:,0:3])

                num_row = label_data.shape[0]
                raw_h5f.set_num_default_row(num_row)
                raw_h5f.append_to_dset('bounding_box', label_data)
                raw_h5f.create_done()


    @staticmethod
    def showfilesummary(file_name):
        h5f_file_name = kitti_prepare.rawh5f_path + file_name
        with h5py.File(h5f_file_name,'r') as h5f:
            show_h5f_summary_info(h5f)


if __name__ == '__main__':
    START_T = time.time()
    min_num = 3000000
    max_num = 0

    benchmark_name = '/KITTI'
    point_cloud_name = 'velodyne'
    point_cloud_file_path = os.path.join(KITTI_DATA_DIR,point_cloud_name)

    point_cloud_file_list = glob.glob(os.path.join(point_cloud_file_path,'*.bin'))
    for pc_f_n in point_cloud_file_list:
        point_cloud_data = read_point_cloud_from_bin(pc_f_n)
        pc_num = point_cloud_data.shape[0]
        if pc_num < min_num:
            min_num = pc_num

        if pc_num > max_num:
            max_num = pc_num


    print('min num is %d, and max num is %d' %(min_num, max_num) )
    #kitti_prepare.generate_kitti_to_rawh5f()
    #kitti_prepare.showfilesummary(file_name)

    T = time.time(()) - START_T
