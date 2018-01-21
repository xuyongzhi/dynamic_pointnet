#------------------------------------------
# created in 20/1/2018, by Xuesong(Ben) LI
# code is for 3D object non_max_suppression
#------------------------------------------


import numpy as np
import shapely.geometry import box, Polygon


def nms_3d(all_boxes, thresh = 0.3):
    '''
    description of algorithm pipeline
        1. sorting all boxes according to score
        2. getting highest score 3d box
        3. comparing the 3D overlap with rest of boxes
        4. deleting the boxes whose overlaps are higher than a threshold
        5. repeatting
    input:
        all_boxes : n x 8(l, w, h,theta, x, y, z, score)
    '''
    assert all_boxes.shape[0] > 0
    # initialize the list of picked indices
    pick = []
    ## sortting the scores
    indices_box = np.argsort(all_boxes[:, 4])

    while(len(indices_box)>0):
        last = len(indices_box) - 1
        i = indices_box[last]
        pick.append(i)

        ## compare i with rest of boxes






def convert_to_list_points(all_boxes):
    # converting the 3D boxes into a list of coordinate point of birdview
    # input shape: n x 8
    # output shape: n list,
    assert all_boxes.shape[0]>0

    all_coordinates = []
    num_point = all_boxes.shape[0]
    for i in range(num_point):
        temp_coordinate = [[],
                           [],
                           [],
                           [],
                           [],
                           []]
