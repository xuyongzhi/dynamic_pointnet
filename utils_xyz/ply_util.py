# xyz
# Jan 2018
import os, sys
import numpy as np
from plyfile import PlyData, PlyElement
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR+'/matterport_metadata')
from get_mpcat40 import MatterportMeta,get_cat40_from_rawcat



def test_plyfile():
    vertex = np.array([(0, 0, 0), (0,1,0), (1,1,0), (1,0,0),
                        (0, 0, 1), (0,1,1), (1,1,1), (1,0,1)],
                        dtype=[('x', 'f8'), ('y', 'f8'),
                               ('z', 'f8')])
    el_vertex = PlyElement.describe(vertex,'vertex')

    #PlyData([el]).write('tmp/test_binary.ply')
    #PlyData([el],text=True).write('tmp/test_ascii.ply')

    face = np.array([([0, 1, 2], 255, 255, 255),
                     ([0, 2, 3], 255,   0,   0),
                     ([0, 1, 3],   0, 255,   0),
                     ([1, 2, 3],   0,   0, 255)],
                    dtype=[('vertex_indices', 'i4', (3,)),
                           ('red', 'u1'), ('green', 'u1'),
                           ('blue', 'u1')])
    el_face = PlyElement.describe(face,'face')

    edge = np.array([(0, 1, 255, 0, 0),
                     (1, 2, 255, 0, 0),
                     (2, 3, 255, 0, 0),
                     (3, 0, 255, 0, 0),
                     (0, 4, 255, 0, 0)],
                    dtype=[('vertex1', 'i4'), ('vertex2','i4'),
                           ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    el_edge = PlyElement.describe(edge,'edge')

    PlyData([el_vertex,el_edge],text=True).write('tmp/test_ascii.ply')
   # PlyData([el],
   #         byte_order='>').write('tmp/big_endian_binary.ply')
    print('write tmp/test_ascii.ply')

def gen_box_pl( ply_fn, box_xyzs, pl_xyz ):
    assert box_xyzs.shape[-1] == 3
    assert pl_xyz.shape[-1] == 3
    assert int(box_xyzs.shape[0]) % 8 == 0

    box_xyzs = np.reshape( box_xyzs, (-1,3) )
    pl_xyz = np.reshape( pl_xyz, (-1,3) )
    num_box = box_xyzs.shape[0] // 8
    num_vertex = box_xyzs.shape[0] + pl_xyz.shape[0]
    vertex = np.zeros( shape=(num_vertex) ).astype([('x', 'f8'), ('y', 'f8'),('z', 'f8')])
    for i in range(box_xyzs.shape[0]):
        vertex[i] = ( box_xyzs[i,0],box_xyzs[i,1],box_xyzs[i,2] )
    for i in range(pl_xyz.shape[0]):
        vertex[i+box_xyzs.shape[0]] = ( pl_xyz[i,0],pl_xyz[i,1],pl_xyz[i,2] )
    el_vertex = PlyElement.describe(vertex,'vertex')

    edge_basic = np.array([(0, 1, 255, 0, 0),
                     (1, 2, 255, 0, 0),
                     (2, 3, 255, 0, 0),
                     (3, 0, 255, 0, 0),
                     (4, 5, 255, 0, 0),
                     (5, 6, 255, 0, 0),
                     (6, 7, 255, 0, 0),
                     (7, 4, 255, 0, 0),
                     (0, 4, 255, 0, 0),
                     (1, 5, 255, 0, 0),
                     (2, 6, 255, 0, 0),
                     (3, 7, 255, 0, 0)] )
    edge_val = np.concatenate( [edge_basic]*num_box,0 )
    for i in range(num_box):
        edge_val[i*12:(i+1)*12,0:2] += (8*i)
    edge = np.zeros( shape=(edge_val.shape[0]) ).astype(
                    dtype=[('vertex1', 'i4'), ('vertex2','i4'),
                           ('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    for i in range(edge_val.shape[0]):
        edge[i] = ( edge_val[i,0], edge_val[i,1], edge_val[i,2], edge_val[i,3], edge_val[i,4] )
    el_edge = PlyElement.describe(edge,'edge')

    PlyData([el_vertex, el_edge],text=True).write(ply_fn)
    print('write %s ok'%(ply_fn))

def test_box( pl_xyz=None ):
    box_xyz0_a = np.array( [(0,0,0),(0,1,0),(1,1,0),(1,0,0)] )
    box_xyz0_b = box_xyz0_a + np.array([0,0,1])
    box_xyz0 = np.concatenate([box_xyz0_a,box_xyz0_b],0)
    box_xyz1 = box_xyz0 + np.array([0.3,0.3,0])
    box_xyz = np.concatenate( [box_xyz0,box_xyz1],0 )

    pl_xyz = box_xyz + np.array([-0.3,-0.3,0.2])

    gen_box_pl( '/tmp/box_pl.ply',box_xyz, pl_xyz )

def create_ply( xyz, ply_fn, label=None, label2color=None, box=None ):
  #  assert xyz.ndim == 3    # (num_block,num_point,3)
  #  assert label.ndim == 2  # (num_block,num_point)
    folder = os.path.dirname(ply_fn)
    if not os.path.exists(folder):
        os.makedirs(folder)
    xyz = np.reshape( xyz,(-1,xyz.shape[-1]) )
    if xyz.shape[-1] == 3 and (label!=None).all() and label2color!=None:
        label2color_ls = []
        for i in range(len(label2color)):
            label2color_ls.append( np.reshape(np.array(label2color[i]),(1,3)) )
        label2colors = np.concatenate( label2color_ls,0 )
        color = np.take( label2colors,label,axis=0 )
        color = np.reshape( color,(-1,3) )
        xyz = np.concatenate([xyz,color],-1)
    if xyz.shape[-1] == 3:
        vertex = np.zeros( shape=(xyz.shape[0]) ).astype([('x', 'f8'), ('y', 'f8'),('z', 'f8')])
        for i in range(xyz.shape[0]):
            vertex[i] = ( xyz[i,0],xyz[i,1],xyz[i,2] )
    if xyz.shape[-1] == 6:
        vertex = np.zeros( shape=(xyz.shape[0]) ).astype([('x', 'f8'), ('y', 'f8'),('z', 'f8'),('red','u1'),('green','u1'),('blue','u1')])
        for i in range(xyz.shape[0]):
            vertex[i] = ( xyz[i,0],xyz[i,1],xyz[i,2],xyz[i,3],xyz[i,4],xyz[i,5] )

    el_vertex = PlyElement.describe(vertex,'vertex')
    PlyData([el_vertex],text=True).write(ply_fn)

    print('save ply file: %s'%(ply_fn))


def create_ply_matterport( xyz, ply_fn, label=None):
    create_ply( xyz,ply_fn,label,MatterportMeta['label2color'] )

if __name__ == '__main__':
    #test_plyfile()
    test_box()

