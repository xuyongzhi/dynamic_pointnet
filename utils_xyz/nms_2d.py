#------------------------------------------
# created in 20/1/2018, by Xuesong(Ben) LI
# code is for 3D object non_max_suppression
#------------------------------------------


import numpy as np
from shapely.geometry import box, Polygon

def nms_2d(all_boxes, thresh = 0.3):
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
    assert all_boxes.shape[1] == 6
    # initialize the list of picked indices
    pick = []
    ## sortting the scores
    indices_box = np.argsort( all_boxes[:, 5])  ## check type of data
    coordinate_list = convert_to_list_points(all_boxes[:,0:5]) # shape: n x 5, data type is list
    while(indices_box.shape[0]>0):
        last = indices_box.shape[0] - 1
        i = indices_box[last]
        pick.append(i)
        if last == 0:
            break
        coordinate_rest_list = [ coordinate_list[indices_box[x]] for x in range(last) ]
        ratio_overlap = caculate_2d_overlap( all_boxes[i,0:5],                  coordinate_list[i],
                                            all_boxes[indices_box[0:last],0:5], coordinate_rest_list )



        indices_box = np.delete(indices_box, np.concatenate((np.array([last]),
                                                             np.where(ratio_overlap>thresh)[0]),axis = 0), axis = 0)


    return all_boxes[pick,:]

def caculate_2d_overlap(cur_box, cur_coordinate, rest_box, rest_coordinate):
    '''
    description:
        input:      cur_box is the l,w,h,alpha,x,y,z   (format is pretty strictly)
                    cur_coordinate is the coordinate list of cur_box
                    rest_box is the array of rest box: n x 8(l,w,h,alpha,x,y,z)
                    rest_coordinate is the list of rest_box
        output:

    '''
    assert rest_box.shape[0]>0
    assert rest_box.shape[1]==5
    assert cur_box.shape[0]==5
    cur_polygon  = Polygon(cur_coordinate)
    if cur_polygon.is_valid==False:
	cur_polygon = cur_polygon.buffer(0)
	
    rest_polygon = []
    num_rest = rest_box.shape[0]
    for j in range( num_rest ):
	p1 = Polygon(rest_coordinate[j])
	if p1.is_valid:
            rest_polygon.append( p1)
	else:
	    rest_polygon.append(p1.buffer(0))

    ## getting birdview overlap firstly
    birdview_overlap = np.array([[ cur_polygon.intersection(rest_polygon[k]).area ] for k in range(num_rest)])
    # 3D bounding box overlap area is birdview_overlap x height_overlap
    area_overlap = birdview_overlap
    cur_box_area = cur_polygon.area
    rest_box_area = np.array([[ rest_polygon[n].area ] for n in range( num_rest )])
    ratio_overlap = area_overlap /np.maximum( cur_box_area + rest_box_area - area_overlap, np.finfo(np.float64).eps) ## np.arry

    return ratio_overlap


def convert_to_list_points(all_boxes):
    # converting the 2D boxes into a list of coordinate point of birdview
    # input shape: n x 5  // 0:l, 1:w, 2:alpha, 3:x, 4:y
    # output shape: n list,
    assert all_boxes.shape[0]>0
    assert all_boxes.shape[1]==5

    all_coordinates = []
    num_point = all_boxes.shape[0]
    for i in range(num_point):
        l = all_boxes[i, 0]
        w = all_boxes[i, 1]
        xy = np.array([[-w/2.0, l/2.0],[w/2.0, l/2.0],[w/2.0, -l/2.0],[-w/2.0, -l/2.0], [-w/2.0, l/2.0]])
        theta = - all_boxes[i, 2]
        T  = np.array([[np.cos(theta), np.sin(theta) ],[-np.sin(theta), np.cos(theta)]])
        a_xy = np.dot(xy, T)
        x_ctr = all_boxes[i, 3]
        y_ctr = all_boxes[i, 4]
        temp_coordinate = a_xy + np.array([[x_ctr, y_ctr]])
        temp_coordinate = temp_coordinate.tolist()  # shape: 5 x 2, list
        all_coordinates.append(temp_coordinate)


    return all_coordinates   # shape: n x 5 x 2



if __name__ == '__main__':
    # generating some data to test wwritten code
    all_boxes = np.random.rand(900, 8)
    nms_boxes = nms_3d(all_boxes)
