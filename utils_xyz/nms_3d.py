#------------------------------------------
# created in 20/1/2018, by Xuesong(Ben) LI
# code is for 3D object non_max_suppression
#------------------------------------------




import numpy as np

def num_3d(all_boxes):
    '''
    description of algorithm pipeline
        1. sorting all boxes according to score
        2. getting highest score 3d box
        3. comparing the 3D overlap with rest of boxes
        4. deleting the boxes whose overlaps are higher than a threshold
        5. repeatting
    input:
        all_boxes : n x 9(index, l, w, h,theta, x, y, z, score)
    '''
    assert all_boxes.shape[0] > 0

    indices_box = np.argsort(all_box[:, 9])

    while(len(indices_box)>0):
