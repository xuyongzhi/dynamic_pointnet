#---------------------------------
# preparing for bounding box regression for 3D region proposls network
# created in 26/12/2017
#----------------------------------

import numpy as np

def 3D_box_transform(anch_boxes, gt_boxes):
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
 
