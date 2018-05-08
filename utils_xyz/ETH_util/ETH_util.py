# May 2018 xyz

import os
import numpy as np


ETH_Meta = {}

ETH_Meta['label2class'] = {0: 'unlabeled points', 1: 'man-made terrain', 2: 'natural terrain',\
                    3: 'high vegetation', 4: 'low vegetation', 5: 'buildings', \
                    6: 'hard scape', 7: 'scanning artefacts', 8: 'cars'}
ETH_Meta['unlabelled_categories'] = [0]
ETH_Meta['easy_categories'] = []
ETH_Meta['label2color'] = \
                {0:	[0,0,0],1:	[0,0,255],2:	[0,255,255],3: [255,255,0],4: [255,0,255],
                6: [0,255,0],7: [170,120,200],8: [255,0,0],5:[10,200,100]}
ETH_Meta['label_names'] = [ETH_Meta['label2class'][l] for l in range(len(ETH_Meta['label2class']))]


def parse_raw_ETH( fn_7z ):
    base_fn = os.path.splitext( fn_7z )[0]
    if base_fn[-3:] == 'txt':
        base_fn = os.path.splitext(base_fn)[0]
    fn_txt = base_fn+'.txt'
    fn_labels = base_fn+'.labels'
    if not os.path.exists( fn_txt ):
        os.system( '7za e %s -o%s'%(fn_7z, os.path.dirname(fn_7z)) )


    # {x, y, z, intensity, r, g, b}
    raw_data = np.loadtxt( fn_txt )
    xyz = raw_data[:,0:3]
    intensity = raw_data[:,3:4]
    rgb = raw_data[:,4:7]
    if os.path.exists( fn_labels ):
        labels = np.loadtxt(fn_labels).reshape( (-1,1) )
        assert xyz.shape[0] == labels.shape[0]
    else:
        #labels = np.ones(shape=(xyz.shape[0],1))*(-111)
        labels = None
    return xyz, intensity, rgb, labels
