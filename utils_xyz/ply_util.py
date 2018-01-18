# xyz
# Jan 2018
import os
import numpy as np
from plyfile import PlyData, PlyElement

def test():
    vertex = np.array([(0, 0, 0),
                         (0, 1, 1),
                         (1, 0, 1),
                         (1, 1, 0)],
                        dtype=[('x', 'f4'), ('y', 'f4'),
                               ('z', 'f4')])
    el = PlyElement.describe(vertex,'vertex')
    PlyData([el]).write('tmp/test_binary.ply')
    PlyData([el],text=True).write('tmp/test_ascii.ply')
    PlyData([el],
            byte_order='>').write('tmp/big_endian_binary.ply')

def create_ply( xyz, ply_fn, label=None,label2color=None ):
  #  assert xyz.ndim == 3    # (num_block,num_point,3)
  #  assert label.ndim == 2  # (num_block,num_point)
    folder = os.path.dirname(ply_fn)
    if not os.path.exists(folder):
        os.makedirs(folder)
    xyz = np.reshape( xyz,(-1,xyz.shape[-1]) )
    if xyz.shape[-1] == 3:
        vertex = np.zeros( shape=(xyz.shape[0]) ).astype([('x', 'f4'), ('y', 'f4'),('z', 'f4')])
        for i in range(xyz.shape[0]):
            vertex[i] = ( xyz[i,0],xyz[i,1],xyz[i,2] )
    if xyz.shape[-1] == 6:
        vertex = np.zeros( shape=(xyz.shape[0]) ).astype([('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red','u1'),('green','u1'),('blue','u1')])
        for i in range(xyz.shape[0]):
            vertex[i] = ( xyz[i,0],xyz[i,1],xyz[i,2],xyz[i,3],xyz[i,4],xyz[i,5] )
    el = PlyElement.describe(vertex,'vertex')
    PlyData([el],text=True).write(ply_fn)
    print('save ply file: %s'%(ply_fn))

if __name__ == '__main__':
    test()

