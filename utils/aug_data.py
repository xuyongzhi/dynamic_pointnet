# May 2018
# xyz

import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

import geometric_util as geo_util

def aug_data_asg( batch_xyz, aug_type ):
    '''
    data augmentation after sampling and grouping
    Inputs:
        batch_xyz: [N,W,3]: N global blocks (batches), W points in each global block

    Notes:
        rotation aug after sampling and grouping can be performed on the fly.
                     before sampling and grouping cannot yet.
        (1) rotation after sampling and grouping:  xyz change, bxmap preserve
        (2) rotation before sampling and grouping: xyz change, bxmap change
    '''
    if aug_type == 'gbr': # global block rotation before sg
        batch_xyz = geo_util.point_rotation_randomly( batch_xyz, rxyz_max = np.pi*np.array([0.1,0.1,2]) )

#def aug_data_bsg( batch_xyz,  ):

