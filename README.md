# pointnet with dynamic sampling, pyramid multiscale architecture and difficulty aware abilities
created by benz, xyz based on fork of Pointnet++

### Urgent codeing issues to be fixed
- [ ] The data preparation time cost is too high. 20 ms each block on NCI. Learning is also to slow, 100 ms.

### Model optimization plans
- [ ] Modify pointnet to dense net architecture
- [ ] Multi-architecture based on pyramid pooling
- [ ] Be difficulty-aware
- [ ] Compare pyramid and stacked feature encodering

### Sampling optimization plans
- [ ] Dynamic pooling based on previous prediction


### Performace items to be improved
- [ ] Each box use fix number points
- [ ] 3D learning on large scale scene: large block size


# The main workflow of our 3D instance segmentation network
* split scene to block [2,2,-1] m
* sample block to 8192 points. 
  Output: [b,8192,6]
* (PFE) point feature encoder: learn the feature of each point with { [1x1 conv]x2 -> block maxpooling -> concatenate to each point } x 5.
  Output: [b,8192,512]   --output: mf = multiscale feature in block
* group to voxel [0.2,0.2,0.2]m, num point = 8
  Output: [b,1000,8,512]
* (VFE) feed each voxel to voxel feature encoder: { [8x1 conv] }
  Out: [b,1000,512]   --output: vf = voxel feature 
* (3DRPN)get object regions from voxels by 3D RPN
* Do semantic segmentation within each region
* Back inference voxel label to each point within a voxel

## PFE
1. Stacked PFE architecture
   just as PointNet++
2. Pyramid PFE architecture
   Assume the first point feature encoder is deep enough. Use pyramid pooling to get multi-scale features.
   Unlike PointNet++ where block feature is only concatenated once, it can be conatenated iteratively like VoxelNet.


# Literature Review problems and ideas
* What is the approach of region mask fusion in MaskRCNN.
* Fusion of voxelnet and PointNet:
  VoxelNet advantage: voxel feature encoder learns the voxel feature iteratively and many times. But PointNet only lears once.
           disadvantage: voxel size is very small, one point can never learn larger scale shape information. But the PointNet block size is much larger.
  -> Generate voxel within a block after feature encoder of PointNet, instead of from the whole scene.
  -> Then feed the voxel to RPN.
* I believe Deformable convolution may be significantly useful for 3D learning.
  Actually, it is a dynamicaly sampling for convulutional locationsbased on learnt features. Theoretically, the learnt offset should be instance region proposal.
  - Let deformabel convolution automatically sample dense in difficult area and sparse in easy area.
  - Or just use deformabel convolution to make region proposal of difficult area. Beacuse the shape of difficult area can definitely be replaced by a box. This technique may be useful.




# From Charles R. Qi
Update with https://github.com/charlesq34/pointnet2/commits/master 
Nov 10, 2017  
ec22300f87411c8896cd2d13fd6f2ebd7ad37e10
### PointNet++: *Deep Hierarchical Feature Learning on Point Sets in a Metric Space*
Created by <a href="http://charlesrqi.com" target="_blank">Charles R. Qi</a>, <a href="http://stanford.edu/~ericyi">Li (Eric) Yi</a>, <a href="http://ai.stanford.edu/~haosu/" target="_blank">Hao Su</a>, <a href="http://geometry.stanford.edu/member/guibas/" target="_blank">Leonidas J. Guibas</a> from Stanford University.

This is work is based on our paper linked <a href="https://arxiv.org/pdf/1706.02413.pdf">here</a>. The code release is still in an ongoing process... Stay tuned!

Current release includes TF operators (CPU and GPU), some core pointnet++ layers and a few example network models.

The TF operators are included under `tf_ops`, you need to compile them (check `tf_xxx_compile.sh` under each ops subfolder) first. Update `nvcc` and `python` path if necessary. The code is tested under TF1.2.0. If you are using earlier version it's possible that you need to remove the `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly.

TF and pointnet++ utility layers are defined under `utils/tf_util.py` and `utils/pointnet_util.py`

Under `models`, two classification models (SSG and MSG) and SSG models for part and semantic segmentation have been included.

#### Point Cloud Data
You can get our sampled point clouds of ModelNet40 (XYZ and normal from mesh, 10k points per shape) at this <a href="https://1drv.ms/u/s!ApbTjxa06z9CgQfKl99yUDHL_wHs">OneDrive link</a>. The ShapeNetPart dataset (XYZ, normal and part labels) can be found <a href="https://1drv.ms/u/s!ApbTjxa06z9CgQnl-Qm6KI3Ywbe1">here</a>.
