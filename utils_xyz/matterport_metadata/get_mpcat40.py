import os,csv
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
mpcat40_fn = BASE_DIR+'/mpcat40.tsv'
category_mapping_fn = BASE_DIR+'/category_mapping.tsv'

def hexcolor2int(hexcolor):
    hexcolor = hexcolor[1:]
    hex0 = '0x'+hexcolor[0:2]
    hex1 = '0x'+hexcolor[2:4]
    hex2 = '0x'+hexcolor[4:6]
    hex_int = [int(hex0,16),int(hex1,16),int(hex2,16)]
    return hex_int

MatterportMeta = {}
MatterportMeta['label2class'] = {}
MatterportMeta['label2color'] = {}
MatterportMeta['label2WordNet_synset_key'] = {}
MatterportMeta['label2NYUv2_40label'] = {}
MatterportMeta['label25'] = {}
MatterportMeta['label26'] = {}

with open(mpcat40_fn,'r') as f:
    reader = csv.reader(f,delimiter='\t')
    for i,line in enumerate(reader):
        if i==0: continue
        k = int(line[0])
        MatterportMeta['label2class'][k] = line[1]
        MatterportMeta['label2color'][k] = hexcolor2int( line[2] )
        MatterportMeta['label2WordNet_synset_key'][k]=line[3]
        MatterportMeta['label2NYUv2_40label'][k] = line[4]
        MatterportMeta['label25'][k] = line[5]
        MatterportMeta['label26'][k] = line[6]

with open(category_mapping_fn,'r') as f:
    '''
    ['index', 'raw_category', 'category', 'count', 'nyuId', 'nyu40id', 'eigen13id', 'nyuClass', 'nyu40class', 'eigen13class',
    'ModelNet40', 'ModelNet10', 'ShapeNetCore55', 'synsetoffset', 'wnsynsetid', 'wnsynsetkey', 'mpcat40index', 'mpcat40']
    '''
    reader = csv.reader(f,delimiter='\t')

    mapping_vals = {}
    for i,line in enumerate(reader):
        if i==0:
            ele_names = line
            #print(ele_names)
            #print('\n')
            for j in range(18):
                mapping_vals[ele_names[j]] = []
        else:
            for j in range(18):
                mapping_vals[ele_names[j]].append( line[j] )
                #print(ele_names[j])
                #print(mapping_vals[ele_names[j]])
    mapping_vals['index'] = [int(v) for v in mapping_vals['index']]
    mapping_vals['mpcat40index'] = [int(v) for v in mapping_vals['mpcat40index']]

    assert mapping_vals['index'] == range(1,1+len(mapping_vals['index']))

    #print(mapping_vals['raw_category'])
    #print(mapping_vals['category'])
    #print(mapping_vals['mpcat40index'])
    #print(mapping_vals['mpcat40'])

rawcategory_2_mpcat40ind = mapping_vals['mpcat40index']
rawcategory_2_mpcat40 = mapping_vals['mpcat40']

def get_cat40_from_rawcat(raw_category_indexs):
    '''
    raw_category_indexs.shape=[batch_size,num_point,1]
    '''
    assert raw_category_indexs.shape[2]==1
    mpcat40_idxs = np.zeros(shape=raw_category_indexs.shape)
    batch_size = raw_category_indexs.shape[0]
    num_point = raw_category_indexs.shape[1]
    mpcat40s = [ ['']*num_point ]*batch_size
    for i in range(batch_size):
        for j in range(num_point):
            raw_category_index = raw_category_indexs[i,j,0]

            #???????????????????????????????????????????????????????????????????
            if raw_category_index==0:
                raw_category_index = 40


            if raw_category_index<=0:
                print('raw_category_index=%d bug should >0'%(raw_category_index))
                print('err num: %d'%(np.sum(raw_category_indexs<=0)))
                import pdb; pdb.set_trace()  # XXX BREAKPOINT
            assert raw_category_index>0, "raw_category_index start from 1"
            mpcat40_ij = rawcategory_2_mpcat40[raw_category_index-1]
            mpcat40_idx_ij = rawcategory_2_mpcat40ind[raw_category_index-1]
            mpcat40_idxs[i,j,0] = mpcat40_idx_ij
            assert mpcat40_ij == MatterportMeta['label2class'][mpcat40_idx_ij],"%s != %s"%(mpcat40_ij,MatterportMeta['label2class'][mpcat40_idx_ij])
            mpcat40s[i][j] += mpcat40_ij
    return mpcat40_idxs

