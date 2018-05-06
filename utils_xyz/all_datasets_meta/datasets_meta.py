# May 2018 xyz

from __future__ import print_function
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)


DATASETS = ['MATTERPORT', 'SCANNET', 'ETH']

for ds in DATASETS:
    sys.path.append('%s/%s_util'%(ROOT_DIR,ds))

from MATTERPORT_util import MATTERPORT_Meta
from SCANNET_util import SCANNET_Meta
from ETH_util import ETH_Meta
DATASETS_Meta = [MATTERPORT_Meta, SCANNET_Meta, ETH_Meta]


class DatasetsMeta():
    g_label2class_dic = {}
    g_label_names_dic = {}
    g_unlabelled_categories = {}
    g_easy_categories_dic = {}
    g_label2color_dic = {}

    for i in range(len(DATASETS)):
        DS_i = DATASETS[i]
        DS_Meta_i = DATASETS_Meta[i]
        g_label2class_dic[DS_i] = DS_Meta_i['label2class']
        g_label_names_dic[DS_i] = DS_Meta_i['label_names']
        g_label2color_dic[DS_i] = DS_Meta_i['label2color']
        g_easy_categories_dic[DS_i] = DS_Meta_i['easy_categories']
        g_unlabelled_categories[DS_i] = DS_Meta_i['unlabelled_categories']
    ##---------------------------------------------------------------------------

    #---------------------------------------------------------------------------
    g_label2class_dic['STANFORD_INDOOR3D'] = \
                    {0:'ceiling', 1:'floor', 2:'wall', 3:'beam', 4:'column', 5:'window', 6:'door', 7:'table',
                     8:'chair', 9:'sofa', 10:'bookcase', 11:'board', 12:'clutter'}
    g_unlabelled_categories['STANFORD_INDOOR3D'] = [12]
    g_easy_categories_dic['STANFORD_INDOOR3D'] = []
    g_label2color_dic['STANFORD_INDOOR3D'] = \
                    {0:	[0,0,0],1:	[0,0,255],2:	[0,255,255],3: [255,255,0],4: [255,0,255],10: [100,100,255],
                    6: [0,255,0],7: [170,120,200],8: [255,0,0],9: [200,100,100],5:[10,200,100],11:[200,200,200],12:[200,200,100]}

    #---------------------------------------------------------------------------
    g_label2class_dic['KITTI'] = {0:'background', 1:'car', 2:'pedestrian', 3:'cyclist'}   ## benz_m
    g_unlabelled_categories['KITTI'] = []
    g_label2color_dic['KITTI'] = { 0:[0,0,0], 1:[0,0,255], 2:[0,255,255], 3:[255,255,0] }     ## benz_m
    g_easy_categories_dic['KITTI'] = []

    def __init__(self,datasource_name):
        self.datasource_name = datasource_name
        self.g_label2class = self.g_label2class_dic[self.datasource_name]
        self.g_label2color = self.g_label2color_dic[self.datasource_name]
        self.g_class2label = {cls:label for label,cls in self.g_label2class.iteritems()}
        self.g_class2color = {}
        for i in self.g_label2class:
            cls = self.g_label2class[i]
            self.g_class2color[cls] = self.g_label2color[i]
        self.num_classes = len(self.g_label2class)
        #self.num_classes = len(self.g_label2class) - len(self.g_unlabelled_categories[self.datasource_name])

