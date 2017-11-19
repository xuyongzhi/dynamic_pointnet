# xyz
# split generated scannet pickle data
import pickle
import os
import sys
import numpy as np

BASE_DIR = os.path.dirname( os.path.abspath(__file__) )
ROOT_DIR = os.path.dirname( BASE_DIR )
DATA_DIR = os.path.join( ROOT_DIR,'data/scannet_data' )


def split_pickle(file_name,split_row=10):
    with open(file_name,'rb') as fp:
        scene_points_list = pickle.load(fp)
        semantic_labels_list = pickle.load(fp)

        num_scans = len(scene_points_list)
        split_row = min(num_scans,split_row)
        scene_points_list = scene_points_list[0:split_row]
        semantic_labels_list = semantic_labels_list[0:split_row]
