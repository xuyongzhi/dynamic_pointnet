# May 2018 xyz

from __future__ import print_function
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR+'/matterport_metadata')
from get_mpcat40 import MatterportMeta,get_cat40_from_rawcat

class DatasetsMeta():
    g_label2class_dic = {}
    g_unlabelled_categories = {}
    g_easy_categories_dic = {}
    g_label2color_dic = {}

    #---------------------------------------------------------------------------
    g_label2class_dic['MATTERPORT'] = MatterportMeta['label2class']
    g_unlabelled_categories['MATTERPORT'] = [0,41]
    g_easy_categories_dic['MATTERPORT'] = []
    g_label2color_dic['MATTERPORT'] = MatterportMeta['label2color']

    #---------------------------------------------------------------------------
    g_label2class_dic['ETH'] = {0: 'unlabeled points', 1: 'man-made terrain', 2: 'natural terrain',\
                     3: 'high vegetation', 4: 'low vegetation', 5: 'buildings', \
                     6: 'hard scape', 7: 'scanning artefacts', 8: 'cars'}
    g_unlabelled_categories['ETH'] = [0]
    g_easy_categories_dic['ETH'] = []
    g_label2color_dic['ETH'] = \
                    {0:	[0,0,0],1:	[0,0,255],2:	[0,255,255],3: [255,255,0],4: [255,0,255],
                    6: [0,255,0],7: [170,120,200],8: [255,0,0],5:[10,200,100]}

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
    g_label2class_dic['SCANNET'] = g_label2class_dic['scannet']   = {0:'unannotated', 1:'wall', 2:'floor', 3:'chair', 4:'table', 5:'desk',\
                                6:'bed', 7:'bookshelf', 8:'sofa', 9:'sink', 10:'bathtub', 11:'toilet',\
                                12:'curtain', 13:'counter', 14:'door', 15:'window', 16:'shower curtain',\
                                17:'refridgerator', 18:'picture', 19:'cabinet', 20:'otherfurniture'}
    g_unlabelled_categories['SCANNET'] = [0]
    g_easy_categories_dic['SCANNET'] = [ 2,1,3,6,11,10,4 ]
    g_label2color_dic['SCANNET'] = \
                    {0:	[0,0,0],1:	[0,0,255],2:	[0,255,255],3: [255,255,0],4: [255,0,255],10: [100,100,255],
                    6: [0,255,0],7: [170,120,200],8: [255,0,0],9: [200,100,100],5:[10,200,100],11:[200,200,200],12:[200,200,100],
                    13: [100,200,200],14: [200,100,200],15: [100,200,100],16: [100,100,200],
                     17:[100,100,100],18:[200,200,200],19:[200,200,100],20:[200,200,100]}

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

