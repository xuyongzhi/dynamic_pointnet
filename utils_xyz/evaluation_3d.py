##-------------------------------------------
## function is to evalute the accuracy of detection result
## creaed by Xuesong(Ben) LI, on 22/1/2018
##-------------------------------------------


import numpy as np
from nms_3d import convert_to_list_points
from nms_3d import caculate_3d_overlap




def average_precsion(recall, precesion):
    aveg_precision = 0.
    for t in np.arange(0., 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.sum(precesion[recall>=t])
        aveg_precision = aveg_precision + p/11.

    return aveg_precision


def evaluation_3d( all_pred_boxes, all_gt_boxes, threshold = 0.5 ):
    '''
    description: process the predication result and get the accuracy
        1. putting all gt_boxes together including the index
        2. sorting all_boxes according to the confidence
        3. checking the every predict_box with gt_box accordingly
        4. sum the tp and np and all the gt_box
    input: all_pred_boxes and all_gt_boxes are lists.
    '''
    # change format of all_pred_boxes into type: index,l, w, h, theta, x, y, z, score
    num_all_boxes = len(all_pred_boxes)
    np_all_boxes = np.zeros([1 ,9])

    for i in range(num_all_boxes):
        temp_index = np.full((all_pred_boxes[i].shape[0],1), i)
        temp_all_  = np.concatenate((temp_index, all_pred_boxes[i]), axis=1)
        np_all_boxes = np.append( np_all_boxes, temp_all_, axis=0)
    np_all_boxes  =  np.delete(np_all_boxes, 0, 0)

    sorted_ind    = np.argsort(-np_all_boxes[:,8])
    sorted_scores = np.sort(-np_all_boxes[:,8])

    # convertting the gt_box into coordinate list
    num_gt = 0
    all_gt_boxes_coordinate = []

    # caculate the cooridnate list of gt_boxes
    for j in range(num_all_boxes):
       num_gt = num_gt + all_gt_boxes[j].shape[0]
       all_gt_boxes_coordinate.append(convert_to_list_points(all_gt_boxes[j]))

    num_ind = sorted_ind.shape[0]
    tp  = np.zeros(num_ind)
    fp  = np.zeros(num_ind)

    for d in range(num_ind):   # index in sorted_ind
        np_index_cur = sorted_ind[d]  # index in np_all_boxes
        gt_index_cur = int( np_all_boxes[ np_index_cur ,0] ) # index in gt_boxes
        coordinate_cur = convert_to_list_points(np_all_boxes[ np_index_cur, 1:8].reshape(1, -1))[0]
        # coordinate_cur_gt = all_gt_boxes_coordinate[ gt_index_cur ]

        ratio_overlap = caculate_3d_overlap( np_all_boxes[np_index_cur,1:8], coordinate_cur, all_gt_boxes[gt_index_cur], all_gt_boxes_coordinate[ gt_index_cur ])

        overlap_max  = np.max( ratio_overlap )

        if overlap_max >= threshold:
            tp[d] = 1
        else:
            fp[d] = 1

    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    recall = tp / float(num_gt)

    # avoid divided by zero in case the first detection matches a difficult
    # ground truth
    precesion = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    aveg_precision = average_precsion(recall, precesion)
    return aveg_precision


if __name__ == '__main__':
    # generating the data
    num_box = 400
    all_pred_boxes  = []
    all_gt_boxes = []
    for i in range(num_box):
        all_pred_boxes.append( np.random.rand( np.random.randint(2,50) ,8))
        all_gt_boxes.append( np.random.rand( np.random.randint(2,50) ,7))
    evaluation_3d(all_pred_boxes,all_gt_boxes)
