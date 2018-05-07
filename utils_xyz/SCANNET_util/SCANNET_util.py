import os
BASE_DIR = os.path.dirname( os.path.abspath(__file__) )

SCANNET_Meta = {}
SCANNET_Meta['label2class'] = {0:'unannotated', 1:'wall', 2:'floor', 3:'chair', 4:'table', 5:'desk',\
                            6:'bed', 7:'bookshelf', 8:'sofa', 9:'sink', 10:'bathtub', 11:'toilet',\
                            12:'curtain', 13:'counter', 14:'door', 15:'window', 16:'shower curtain',\
                            17:'refridgerator', 18:'picture', 19:'cabinet', 20:'otherfurniture'}
SCANNET_Meta['unlabelled_categories'] = [0]
SCANNET_Meta['easy_categories'] = [ 2,1,3,6,11,10,4 ]

SCANNET_Meta['label2color'] = \
                {0:	[0,0,0],1:	[0,0,255],2:	[0,255,255],3: [255,255,0],4: [255,0,255],10: [100,100,255],
                6: [0,255,0],7: [170,120,200],8: [255,0,0],9: [200,100,100],5:[10,200,100],11:[200,200,200],12:[200,200,100],
                13: [100,200,200],14: [200,100,200],15: [100,200,100],16: [100,100,200],
                    17:[100,100,100],18:[200,200,200],19:[200,200,100],20:[200,200,100]}

SCANNET_Meta['label_names'] = [SCANNET_Meta['label2class'][l] for l in range(len(SCANNET_Meta['label2class'])) ]

def get_raw2scannet_label_map():
    lines = [line.rstrip() for line in open(BASE_DIR+'/scannet-labels.combined.tsv')]
    lines = lines[1:]
    raw2scannet = {}
    for i in range(len(lines)):
        label_classes_set = set(SCANNET_Meta['label_names'] )
        elements = lines[i].split('\t')
        raw_name = elements[0]
        nyu40_name = elements[6]
        if nyu40_name not in label_classes_set:
            raw2scannet[raw_name] = 'unannotated'
        else:
            raw2scannet[raw_name] = nyu40_name
    return raw2scannet


g_raw2scannet = get_raw2scannet_label_map()



def parse_raw( scene_name ):
    scene_name_base = os.path.basename( scene_name )
    ply_fn = scene_name + '/%s_vh_clean.ply'%(scene_name_base)
    mesh_segs_fn = scene_name + '/%s_vh_clean.segs.json'%(scene_name_base)
    aggregation_fn = scene_name + '/%s_vh_clean.aggregation.json'%(scene_name_base)

    segid_to_pointid, mesh_labels = parse_mesh_segs(mesh_segs_fn)
    points = parse_scan_ply( ply_fn )
    instance_segids, labels = parse_aggregation( aggregation_fn )

    # Each instance's points
    instance_points_list = []
    instance_labels_list = []
    semantic_labels_list = []
    for i in range(len(instance_segids)):
        segids = instance_segids[i]
        pointids = []
        for segid in segids:
            pointids += segid_to_pointid[segid]
        instance_points = points[np.array(pointids),:]
        instance_points_list.append(instance_points)
        instance_labels_list.append(np.ones((instance_points.shape[0], 1))*i)
        if labels[i] not in g_raw2scannet:
            label = 'unannotated'
        else:
            label = g_raw2scannet[labels[i]]
        label = CLASS_NAMES.index(label)
        semantic_labels_list.append(np.ones((instance_points.shape[0], 1))*label)

    # Refactor data format
    scene_points = np.concatenate(instance_points_list, 0)
    instance_labels = np.concatenate(instance_labels_list, 0)
    semantic_labels = np.concatenate(semantic_labels_list, 0)

    return scene_points, instance_labels, semantic_labels, mesh_labels


def WriteRawH5f_SCANNET( scene_name, rawh5f_dir ):
    # save as rh5
    scene_name_base = os.path.basename( scene_name )
    rawh5f_fn = os.path.join(rawh5f_dir, scene_name_base+'.rh5')
    if Raw_H5f.check_rh5_intact( rawh5f_fn )[0]:
        print('rh5 intact: %s'%(rawh5f_fn))
        return scene_name
    print('start write rh5: %s'%(rawh5f_fn))

    scene_points, instance_labels, semantic_labels, mesh_labels = scannet_util.parse_raw( scene_name )
    num_points = scene_points.shape[0]
    with h5py.File(rawh5f_fn,'w') as h5f:
        raw_h5f = Raw_H5f(h5f,rawh5f_fn,'SCANNET')
        raw_h5f.set_num_default_row(num_points)
        raw_h5f.append_to_dset('xyz', scene_points[:,0:3])
        raw_h5f.append_to_dset('color', scene_points[:,3:6])
        raw_h5f.append_to_dset('label_category', semantic_labels)
        raw_h5f.append_to_dset('label_instance', instance_labels)
        raw_h5f.append_to_dset('label_mesh', mesh_labels)
        raw_h5f.rh5_create_done()
    return scene_name

