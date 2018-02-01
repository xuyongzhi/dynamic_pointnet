#------------------------------------------
# created in 20/1/2018, by Xuesong(Ben) LI
# code is for 3D object non_max_suppression
#------------------------------------------


import numpy as np
from shapely.geometry import box, Polygon

def nms_3d(all_boxes, thresh = 0.3):
    '''
    description of algorithm pipeline
        1. sorting all boxes according to score
        2. getting highest score 3d box
        3. comparing the 3D overlap with rest of boxes
        4. deleting the boxes whose overlaps are higher than a threshold
        5. repeatting
    input:
        all_boxes : n x 8(l, w, h,theta, x, y, z, score,.....) make sure that the first 7 elements must be the same format
    '''
    assert all_boxes.shape[0] > 0
    assert all_boxes.shape[1] >= 7
    # initialize the list of picked indices
    pick = []
    ## sortting the scores
    indices_box = np.argsort( all_boxes[:, 7])  ## check type of data
    coordinate_list = convert_to_list_points(all_boxes[:,0:7]) # shape: n x 5, data type is list
    while(indices_box.shape[0]>0):
        last = indices_box.shape[0] - 1
        i = indices_box[last]
        pick.append(i)
        if last == 0:
            break
        '''
        ## compare i with rest of boxes
        current_box = Polygon(coordinate_list[i])
        rest_box = []
        for j in range(last):
            rest_box.append(Polygon(coordinate_list[indices_box[j]]))

        ## getting birdview overlap firstly
        birdview_overlap = np.array([[ current_box.intersection(rest_box[k]) ] for k in range(last)])

        ## height_overlaps = 0,                       if |z1-z2| >= (h1+h2)/2
        ##                 =|(h1+h2)/2 - |z1_z2| |,   if (h1+h2)/2 > |z1 - z2|
        ##                 =h1/2 + h2/2 - (|h2-h1|/2),if |h1-h2|/2 >= |z1 - z2|
        height_overlap  = np.zeros(( birdview_overlap.shape[0], 1))
        cond_1   = np.where((all_boxes[i, 2]+ all_boxes[indices_box[0:last], 2])/2 >= np.abs(all_boxes[i, 6] - all_boxes[indices_box[0:last], 6]) )[0]
        value_1  = ( all_boxes[i, 2]+ all_boxes[indices_box[0:last], 2] )/2 - np.abs( all_boxes[i, 6] - all_boxes[indices_box[0:last], 6] )
        height_overlap[cond_1] = value_1[cond_1]
        cond_2   = np.where( np.abs(all_boxes[i, 2] - all_boxes[indices_box[0:last], 2])/2 > np.abs(all_boxes[i, 6] - all_boxes[indices_box[0:last], 6]) )[0]
        value_2  = ( all_boxes[i, 2]+ all_boxes[indices_box[0:last], 2] )/2 - np.abs(all_boxes[i, 2] - all_boxes[indices_box[0:last], 2])/2
        height_overlap[cond_2] = value_2[cond_2]
        # 3D bounding box overlap area is birdview_overlap x height_overlap
        area_overlap = birdview_overlap*height_overlap
        current_box_area = current_box.area * all_boxes[i, 2]
        rest_box_area = np.array([[ rest_box[n].area * all_boxes[ indices_box[n] ,2] ] for n in range(last)])
        ratio_overlap = area_overlap/(current_box_area + rest_box_area - area) ## np.arry

        # birdview overlap
        # area_overlap = birdview_overlap
        # current_box_area = current_box.area
        # rest_box_area = np.array([[ rest_box[n].area ] for n in range(last)])
        # ratio_overlap = area_overlap/(current_box_area + rest_box_area - area) ## np.arry
        '''
        coordinate_rest_list = [ coordinate_list[indices_box[x]] for x in range(last) ]
        ratio_overlap = caculate_3d_overlap( all_boxes[i,0:7],                  coordinate_list[i],
                                            all_boxes[indices_box[0:last],0:7], coordinate_rest_list )



        indices_box = np.delete(indices_box, np.concatenate((np.array([last]),
                                                             np.where(ratio_overlap>thresh)[0]),axis = 0), axis = 0)


    return all_boxes[pick,:]

def caculate_3d_overlap(cur_box, cur_coordinate, rest_box, rest_coordinate):
    '''
    description:
        input:      cur_box is the l,w,h,alpha,x,y,z   (format is pretty strictly)
                    cur_coordinate is the coordinate list of cur_box
                    rest_box is the array of rest box: n x 8(l,w,h,alpha,x,y,z)
                    rest_coordinate is the list of rest_box
        output:

    '''
    assert rest_box.shape[0]>0
    assert rest_box.shape[1]==7
    assert cur_box.shape[0]==7
    cur_polygon  = Polygon(cur_coordinate)
    rest_polygon = []
    num_rest = rest_box.shape[0]
    for j in range( num_rest ):
        rest_polygon.append(Polygon( rest_coordinate[j] ))

    ## getting birdview overlap firstly
    birdview_overlap = np.array([[ cur_polygon.intersection(rest_polygon[k]).area ] for k in range(num_rest)])
    ## height_overlaps = 0,                       if |z1-z2| >= (h1+h2)/2
    ##                 =|(h1+h2)/2 - |z1_z2| |,   if (h1+h2)/2 > |z1 - z2|
    ##                 =h1/2 + h2/2 - (|h2-h1|/2),if |h1-h2|/2 >= |z1 - z2|
    height_overlap  = np.zeros(( birdview_overlap.shape[0], 1))
    cond_1   = np.where( (cur_box[2] + rest_box[:,2])/2. >= np.abs( cur_box[6] - rest_box[:,6]))[0]

    value_1  = np.abs(( cur_box[2] + rest_box[:,2])/2. - np.abs( cur_box[6] - rest_box[:,6]))
    value_1  = value_1.reshape(-1,1)
    height_overlap[cond_1] = value_1[cond_1]
    cond_2   = np.where( np.abs( cur_box[2] - rest_box[:,2] )/2. > np.abs( cur_box[6] - rest_box[:,6]))[0]
    value_2  = ( cur_box[2] + rest_box[:, 2] )/2. - np.abs( cur_box[2] - rest_box[:, 2])/2.
    value_2  = value_2.reshape(-1,1)
    height_overlap[cond_2] = value_2[cond_2]
    # 3D bounding box overlap area is birdview_overlap x height_overlap
    area_overlap = birdview_overlap*height_overlap
    cur_box_area = cur_polygon.area * cur_box[2]
    rest_box_area = np.array([[ rest_polygon[n].area * rest_box[n,2] ] for n in range( num_rest )])
    ratio_overlap = area_overlap /np.maximum( cur_box_area + rest_box_area - area_overlap, np.finfo(np.float64).eps) ## np.arry

    return ratio_overlap


def convert_to_list_points(all_boxes):
    # converting the 3D boxes into a list of coordinate point of birdview
    # input shape: n x 8
    # output shape: n list,
    assert all_boxes.shape[0]>0
    assert all_boxes.shape[1]==7

    all_coordinates = []
    num_point = all_boxes.shape[0]
    for i in range(num_point):
        # after rotation, x = cos(theta)*x0 - sin(theta)*y0
        # y = sin(theta)*x0 + cos(theta)*y0
        l = all_boxes[i, 0]
        w = all_boxes[i, 1]
        xy = np.array([[-w/2.0, l/2.0],[w/2.0, l/2.0],[w/2.0, -l/2.0],[-w/2.0, -l/2.0], [-w/2.0, l/2.0]])
        theta = - all_boxes[i, 3]
        T  = np.array([[np.cos(theta), np.sin(theta) ],[-np.sin(theta), np.cos(theta)]])
        a_xy = np.dot(xy, T)
        x_ctr = all_boxes[i, 4]
        y_ctr = all_boxes[i, 5]
        temp_coordinate = a_xy + np.array([[x_ctr, y_ctr]])
        temp_coordinate = temp_coordinate.tolist()  # shape: 5 x 2, list
        all_coordinates.append(temp_coordinate)


    return all_coordinates   # shape: n x 5 x 2



if __name__ == '__main__':
    # generating some data to test wwritten code
    all_boxes = np.random.rand(900, 8)
    nms_boxes = nms_3d(all_boxes)
