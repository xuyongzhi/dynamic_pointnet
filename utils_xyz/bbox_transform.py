#---------------------------------
# preparing for bounding box regression for 3D region proposls network
# created in 26/12/2017
#----------------------------------

import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../config'))
import numpy as np



def bbox_transform(anch_boxes, gt_boxes):
    # getting the l,w,h,alpha,x,y,z from anchors
    anch_lengths = anch_boxes[:, 0]
    anch_widths  = anch_boxes[:, 1]
    anch_heights = anch_boxes[:, 2]
    anch_alpha   = anch_boxes[:, 3]
    anch_ctr_x   = anch_boxes[:, 4]
    anch_ctr_y   = anch_boxes[:, 5]
    anch_ctr_z   = anch_boxes[:, 6]
    # getting the l,w,h,alpha,x,y,z from ground truths
    gt_lengths   = gt_boxes[:, 0]
    gt_widths    = gt_boxes[:, 1]
    gt_heights   = gt_boxes[:, 2]
    gt_alpha     = gt_boxes[:, 3]
    gt_ctr_x     = gt_boxes[:, 4]
    gt_ctr_y     = gt_boxes[:, 5]
    gt_ctr_z     = gt_boxes[:, 6]

    targets_dx   = ( gt_ctr_x - anch_ctr_x ) / anch_lengths
    targets_dy   = ( gt_ctr_y - anch_ctr_y ) / anch_widths
    targets_dz   = ( gt_ctr_z - anch_ctr_z ) / anch_heights
    targets_dl   = np.log( gt_lengths / anch_lengths )
    targets_dw   = np.log( gt_widths  / anch_widths  )
    targets_dh   = np.log( gt_heights / anch_heights )
    targets_alpha= ( gt_alpha - anch_alpha ) / (np.pi/4)

    targets      = np.vstack(
	(targets_dl, targets_dw, targets_dh, targets_alpha, targets_dx, targets_dy, targets_dz)).transpose()

    return targets



def bbox_transform_inv(pred_box, xyz):
    ##
    num_class   = cfg.TRAIN.NUM_CLASSES
    num_regression = cfg.TRAIN.NUM_REGRESSION
    num_anchors = cfg.TRAIN.NUM_ANCHORS
    anchor_length = cfg.TRAIN.Anchors[0]
    anchor_width  = cfg.TRAIN.Anchors[1]
    anchor_height = cfg.TRAIN.Anchors[2]
    anchor_alpha  = cfg.TRAIN.Alpha
    temp_pred_box_l   = np.array([ np.exp( pred_box[:,(x*num_regression)])*anchor_length  for x in range(num_anchors)])
    temp_pred_box_l   = temp_pred_box_l.reshape(-1,1)
    temp_pred_box_w   = np.array([ np.exp( pred_box[:,(x*num_regression+1)])*anchor_width  for x in range(num_anchors)])
    temp_pred_box_w   = temp_pred_box_w.reshape(-1,1)
    temp_pred_box_h   = np.array([ np.exp( pred_box[:,(x*num_regression+2)])*anchor_height  for x in range(num_anchors)])
    temp_pred_box_h   = temp_pred_box_h.reshape(-1,1)
    temp_pred_box_alpha = np.array([ pred_box[:,(x*num_regression+3)]*np.pi/4+anchor_alpha[x,0]  for x in range(num_anchors)])
    temp_pred_box_alpha = temp_pred_box_alpha.reshape(-1,1)
    temp_pred_box_x   = np.array([ pred_box[:,(x*num_regression+4)]*anchor_length + xyz[:,0]  for x in range(num_anchors) ])
    temp_pred_box_x   = temp_pred_box_x.reshape(-1,1)
    temp_pred_box_y   = np.array([ pred_box[:,(x*num_regression+5)]*anchor_width + xyz[:,1]  for x in range(num_anchors) ])
    temp_pred_box_y   = temp_pred_box_y.reshape(-1,1)
    temp_pred_box_z   = np.array([ pred_box[i][j,:,(x*num_regression+6)]*anchor_height + xyz[:,2]  for x in range(num_anchors) ])
    temp_pred_box_z   = temp_pred_box_z.reshape(-1,1)

    all_box = np.hstack((temp_pred_box_l, temp_pred_box_w, temp_pred_box_h, temp_pred_box_alpha, temp_pred_box_x, temp_pred_box_y, temp_pred_box_z))

    return all_box
