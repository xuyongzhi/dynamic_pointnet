import pickle
import os
import sys
import numpy as np
import pc_util
import scene_util

class ScannetDataset():
    '''
    (1) Get 1 sub-blocks from each scan.
    (2) The center of each sub-block is randomly selected.
    (3) Sub-block size: [1.5,1.5,:]
    (4) point_set is choiced with 0.2 m padding,
    but sample_weight for padded points are 0.
    '''
    def __init__(self, root, npoints=8192, split='train',small_affix=''):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_filename = os.path.join(self.root, 'scannet_%s%s.pickle'%(split,small_affix))
        with open(self.data_filename,'rb') as fp:
            self.scene_points_list = pickle.load(fp)
            self.semantic_labels_list = pickle.load(fp)
        if split=='train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        elif split=='test':
            self.labelweights = np.ones(21)
    def __getitem__(self, index):
        point_set = self.scene_points_list[index] # (138001, 3)
        semantic_seg = self.semantic_labels_list[index].astype(np.int32) # (138001,)
        coordmax = np.max(point_set,axis=0)
        coordmin = np.min(point_set,axis=0)
        # limit the scope maximum to [1.5,1.5,:] (not used)
        smpmin = np.maximum(coordmax-[1.5,1.5,3.0], coordmin)
        smpmin[2] = coordmin[2]
        smpsz = np.minimum(coordmax-smpmin,[1.5,1.5,3.0])
        smpsz[2] = coordmax[2]-coordmin[2]
        isvalid = False
        for i in range(10):
            # the center of each sub-block is randomly selected
            curcenter = point_set[np.random.choice(len(semantic_seg),1)[0],:]
            # the xy scope of each sub-block is 1.5 m
            curmin = curcenter-[0.75,0.75,1.5]
            curmax = curcenter+[0.75,0.75,1.5]
            curmin[2] = coordmin[2]
            curmax[2] = coordmax[2]
            # 0.2 m padding for sub-block points
            curchoice = np.sum((point_set>=(curmin-0.2))*(point_set<=(curmax+0.2)),axis=1)==3
            cur_point_set = point_set[curchoice,:]  # (15209, 3)
            cur_semantic_seg = semantic_seg[curchoice]
            if len(cur_semantic_seg)==0:
                continue
            # mask idx for mask points within the padded sub-blcok points
            mask = np.sum((cur_point_set>=(curmin-0.01))*(cur_point_set<=(curmax+0.01)),axis=1)==3
            # get the unique 3D volex idx for all mask points
            # for the mask points, normalize to 0~1, then expand to [31,31,62]
            vidx = np.ceil((cur_point_set[mask,:]-curmin)/(curmax-curmin)*[31.0,31.0,62.0])
            vidx = np.unique(vidx[:,0]*31.0*62.0+vidx[:,1]*62.0+vidx[:,2])
            # valid (1): unzero label rate > 0.7
            # valid (2): valid point density > 0.02
            isvalid = np.sum(cur_semantic_seg>0)/len(cur_semantic_seg)>=0.7 and len(vidx)/31.0/31.0/62.0>=0.02
            if isvalid:
                break
        choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
        point_set = cur_point_set[choice,:]
        semantic_seg = cur_semantic_seg[choice]
        mask = mask[choice]
        sample_weight = self.labelweights[semantic_seg]
        sample_weight *= mask
        # (8192,3) (8192,) (8192,)
        return point_set, semantic_seg, sample_weight
    def __len__(self):
        return len(self.scene_points_list)

class ScannetDatasetWholeScene():
    '''
    () Sliding by x y to get sub-blocks in each scan.
    () Sub-block size: [1.5,1.5,:]
    () point_set is choiced with 0.2 m padding,
    but sample_weight for padded points are 0.
    () For each block, randomly select self.npoint points from the padded points.
    '''
    def __init__(self, root, npoints=8192, split='train',small_affix=''):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_filename = os.path.join(self.root, 'scannet_%s%s.pickle'%(split,small_affix))
        with open(self.data_filename,'rb') as fp:
            import gc
            gc.disable()
            self.scene_points_list = pickle.load(fp)
            self.semantic_labels_list = pickle.load(fp)
            gc.enable()
        if split=='train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        elif split=='test':
            self.labelweights = np.ones(21)
    def __getitem__(self, index):
        # the raw data of a scan
        point_set_ini = self.scene_points_list[index] # (138001, 3)
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32) #(138001, coordmax)
        # get the scope of this scan
        coordmax = np.max(point_set_ini,axis=0) # [ 7.08070803,  7.04894161,  3.17538071]
        coordmin = np.min(point_set_ini,axis=0) # [ 0.03255994,  0.22046287,  0.66974348]
        # the stride number
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/1.5).astype(np.int32) # 5
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/1.5).astype(np.int32) # 5
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        isvalid = False
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                # the scope of cur sub-block
                curmin = coordmin+[i*1.5,j*1.5,0]
                curmax = coordmin+[(i+1)*1.5,(j+1)*1.5,coordmax[2]-coordmin[2]]
                # get the indexs of points belong to this sub-block
                # use 0.2 padding scope
                curchoice = np.sum((point_set_ini>=(curmin-0.2))*(point_set_ini<=(curmax+0.2)),axis=1)==3
                cur_point_set = point_set_ini[curchoice,:]
                cur_semantic_seg = semantic_seg_ini[curchoice]
                if len(cur_semantic_seg)==0:
                    continue
                # use 0 padding scope, get if the point is the mask
                mask = np.sum((cur_point_set>=(curmin-0.001))*(cur_point_set<=(curmax+0.001)),axis=1)==3
                # select self.npoints from cur_semantic_seg
                # replace = True, so cur_semantic_seg can < self.npoints
                choice = np.random.choice(len(cur_semantic_seg), self.npoints, replace=True)
                point_set = cur_point_set[choice,:] # Nx3
                semantic_seg = cur_semantic_seg[choice] # N
                mask = mask[choice]
                if sum(mask)/float(len(mask))<0.01: # the rate of mask points
                    continue
                # get the sample weight for all selected points
                sample_weight = self.labelweights[semantic_seg]
                # the sample weight for padded points are set 0
                sample_weight *= mask # N
                point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
                semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
                sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN
        point_sets = np.concatenate(tuple(point_sets),axis=0)   #(22, 8192, 3)
        semantic_segs = np.concatenate(tuple(semantic_segs),axis=0) #(8192, 3)
        sample_weights = np.concatenate(tuple(sample_weights),axis=0) #(8192, 3)
        return point_sets, semantic_segs, sample_weights
    def __len__(self):
        return len(self.scene_points_list)

class ScannetDatasetVirtualScan():
    def __init__(self, root, npoints=8192, split='train'):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_filename = os.path.join(self.root, 'scannet_%s.pickle'%(split))
        with open(self.data_filename,'rb') as fp:
            self.scene_points_list = pickle.load(fp)
            self.semantic_labels_list = pickle.load(fp)
        if split=='train':
            labelweights = np.zeros(21)
            for seg in self.semantic_labels_list:
                tmp,_ = np.histogram(seg,range(22))
                labelweights += tmp
            labelweights = labelweights.astype(np.float32)
            labelweights = labelweights/np.sum(labelweights)
            self.labelweights = 1/np.log(1.2+labelweights)
        elif split=='test':
            self.labelweights = np.ones(21)
    def __getitem__(self, index):
        point_set_ini = self.scene_points_list[index]
        semantic_seg_ini = self.semantic_labels_list[index].astype(np.int32)
        sample_weight_ini = self.labelweights[semantic_seg_ini]
        point_sets = list()
        semantic_segs = list()
        sample_weights = list()
        for i in xrange(8):
            smpidx = scene_util.virtual_scan(point_set_ini,mode=i)
            if len(smpidx)<300:
                continue
            point_set = point_set_ini[smpidx,:]
            semantic_seg = semantic_seg_ini[smpidx]
            sample_weight = sample_weight_ini[smpidx]
            choice = np.random.choice(len(semantic_seg), self.npoints, replace=True)
            point_set = point_set[choice,:] # Nx3
            semantic_seg = semantic_seg[choice] # N
            sample_weight = sample_weight[choice] # N
            point_sets.append(np.expand_dims(point_set,0)) # 1xNx3
            semantic_segs.append(np.expand_dims(semantic_seg,0)) # 1xN
            sample_weights.append(np.expand_dims(sample_weight,0)) # 1xN
        point_sets = np.concatenate(tuple(point_sets),axis=0)
        semantic_segs = np.concatenate(tuple(semantic_segs),axis=0)
        sample_weights = np.concatenate(tuple(sample_weights),axis=0)
        return point_sets, semantic_segs, sample_weights
    def __len__(self):
        return len(self.scene_points_list)


def Cut_Scannet(cut_rate=0.03):
    data_root = os.path.join(DATA_DIR,'scannet_data')
    for split in ['test','train']:
        file_name = data_root+'/scannet_%s.pickle'%(split)
        file_name_new = data_root+'/scannet_%s_small.pickle'%(split)
        with open(file_name,'rb') as fo, open(file_name_new,'wb') as fo_new:
            scene_points_list0 = pickle.load(fo)
            semantic_labels_list0 = pickle.load(fo)
            scene_points_list1 = scene_points_list0[0:int(len(scene_points_list0)*cut_rate)]
            semantic_labels_list1 = semantic_labels_list0[0:int(len(semantic_labels_list0)*cut_rate)]
            pickle.dump(scene_points_list1,fo_new)
            pickle.dump(semantic_labels_list1,fo_new)
            print('gen %s OK'%(file_name_new))

def test():
    #d = ScannetDatasetWholeScene(root = '../data/scannet_data', split='test', npoints=8192,small_affix='_small')
    d = ScannetDataset(root = '../data/scannet_data', split='test', npoints=8192,small_affix='_small')
    labelweights_vox = np.zeros(21)
    for ii in xrange(len(d)):
        print(ii)
        ps,seg,smpw = d[ii]
        print(ps.shape)
        print(ps[0,0:2,:])
        break
        for b in xrange(ps.shape[0]):
            _, uvlabel, _ = pc_util.point_cloud_label_to_surface_voxel_label_fast(ps[b,smpw[b,:]>0,:], seg[b,smpw[b,:]>0], res=0.02)
            tmp,_ = np.histogram(uvlabel,range(22))
            labelweights_vox += tmp
    print labelweights_vox[1:].astype(np.float32)/np.sum(labelweights_vox[1:].astype(np.float32))
    exit()

def PreSampleScannet():
    dataset = ScannetDatasetWholeScene(root = '../data/scannet_data', split='test', npoints=8192,small_affix='_small')



if __name__=='__main__':
    pass
