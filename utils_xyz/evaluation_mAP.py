#--------------------------------------
#This is the evaluation code for 3D point cloud object detection
#created by Xuesong(Ben) Li, 19/1/2018
#-------------------------------------


import numpy as np



def evaluation_mAP():
    '''
    pipeline of evaluation algorithm:
        1. put all pred_box together
        2. collect all the ground truth boxes according to name in a list
        3. sort all pred_box according to confidence
        4. get point cloud data name of every pred_box
        5. compare every pred_box with ground truth box, using point cloud data name to find corresponding ground truth boxes
        6. obtain FP and TP
        7. do np.cumsum(FP) and np.cumsum(TP) to estimate precision and recall curve, and output mAP
    '''
