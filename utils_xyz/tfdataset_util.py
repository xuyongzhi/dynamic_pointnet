# xyz May 2018
import tensorflow as tf
import time
import dataset_utils
from block_data_prep_util import Normed_H5f, Sorted_H5f
from configs import NETCONFIG
import numpy as np

def save_pl_spbin( sh5f, pl_spbin_filename, gsbb_write, S_H5f, IsShowSummaryFinished, data_aug_configs):
    global_num_point = gsbb_write.global_num_point
    assert global_num_point >= gsbb_write.max_global_num_point, "max_global_num_point=%d pl_sph5 file not exist, cannot add global_num_point=%d"%(gsbb_write.max_global_num_point,global_num_point)
    print('start gen sph5 file: ',pl_spbin_filename)
    t0 = time.time()

    with tf.python_io.TFRecordWriter( pl_spbin_filename ) as tfrecord_writer:
        global_attrs = gsbb_write.get_new_attrs('global')

        file_datas = []
        file_global_sample_rate = []
        file_rootb_split_idxmaps = []
        global_sampling_meta_sum = {}
        file_gbixyzs = []

        all_sorted_global_bids = gsbb_write.get_all_sorted_aimbids('global')
        num_global_block_abandoned = 0
        num_point_abandoned = 0

        datasource_name = S_H5f.h5f.attrs['datasource_name']
        sh5f.update_del_labels( data_aug_configs, datasource_name )
        for i,global_block_id in enumerate(all_sorted_global_bids):
            if (i+1) % 30 == 0:
                print('sph5 global_block:%d, abandon %d, total %d'%(i+1, num_global_block_abandoned, all_sorted_global_bids.size) )

            block_datas, data_idxs, rootb_split_idxmap, global_sampling_meta, global_sample_rate = \
                sh5f.get_data_larger_block( global_block_id, gsbb_write, gsbb_write.global_num_point, Normed_H5f.max_rootb_num, data_aug_configs )
            global_bixyz = Sorted_H5f.block_index_to_ixyz_( global_block_id, global_attrs )
            if NETCONFIG['max_global_sample_rate']!=None and  global_sample_rate > NETCONFIG['max_global_sample_rate']:
                num_global_block_abandoned += 1
                num_point_abandoned += block_datas.shape[0]
                continue    # too less points, abandon
            if block_datas.size == 0:
                continue

            file_datas.append(np.expand_dims(block_datas,axis=0))
            file_global_sample_rate.append( global_sample_rate )
            file_gbixyzs.append(np.expand_dims(global_bixyz,axis=0))
            file_rootb_split_idxmaps.append(np.expand_dims(rootb_split_idxmap,axis=0))
            if len( global_sampling_meta_sum ) == 0:
                global_sampling_meta_sum = global_sampling_meta
            else:
                for key in global_sampling_meta:
                    global_sampling_meta_sum[key] += global_sampling_meta[key]


        if len(file_datas) == 0:
            h5f.attrs['intact_void_file'] = 1
            print('all point in this file are void : %s\n'%(pl_spbin_filename))
        else:
            file_datas = np.concatenate(file_datas,axis=0)
            file_global_sample_rate = np.array( file_global_sample_rate )
            file_gbixyzs = np.concatenate(file_gbixyzs,axis=0)
            file_rootb_split_idxmaps = np.concatenate(file_rootb_split_idxmaps,axis=0)

            # For segmentation datasets, the label is inside file_datas
            # For classification datasets, the label has to be extracted by
            # other ways.
            if datasource_name == 'MODELNET40':
                #h5f.attrs['label_category'] = 0
                the_label = Sorted_H5f.extract_label_from_name( pl_spbin_filename, datasource_name )
                object_label = np.reshape( the_label, (1,1,1) )
            else:
                object_label = None
            if datasource_name == 'KITTI':
                g_xyz_center, g_xyz_bottom, g_xyz_top = Sorted_H5f.ixyz_to_xyz( file_gbixyzs, global_attrs )
                import KITTI_util
                file_bounding_boxs, file_datas, file_gbixyzs, file_rootb_split_idxmaps = KITTI_util.extract_bounding_box( pl_spbin_filename, g_xyz_center, g_xyz_bottom, g_xyz_top,\
                                                                                 file_datas, file_gbixyzs, file_rootb_split_idxmaps)
                pl_sph5f.append_to_dset( 'bounding_box', file_bounding_boxs )
                if file_datas.size == 0:
                    h5f.attrs['intact_void_file'] = 1


            example = dataset_utils.pointcloud_to_tfexample(
                S_H5f.h5f.attrs['datasource_name'], file_datas, data_idxs, object_label )
            import pdb; pdb.set_trace()  # XXX BREAKPOINT


            #pl_sph5f.append_to_dset('data',file_datas)
            #pl_sph5f.append_to_dset('block_sample_rate',file_global_sample_rate)

            #if file_labels.size > 0:
            #    pl_sph5f.append_to_dset('labels',file_labels,IsLabelWithRawCategory=False)
            #pl_sph5f.append_to_dset('gbixyz',file_gbixyzs)
            #pl_sph5f.append_to_dset('rootb_split_idxmap', file_rootb_split_idxmaps)
            #for key in global_sampling_meta_sum:
            #    h5f['rootb_split_idxmap'].attrs[key] = global_sampling_meta_sum[key]
            #h5f['rootb_split_idxmap'].attrs['num_global_block_abandoned'] = num_global_block_abandoned
            #h5f['rootb_split_idxmap'].attrs['num_point_abandoned'] = num_point_abandoned
            #h5f['rootb_split_idxmap'].attrs['max_global_sample_rate'] = NETCONFIG['max_global_sample_rate']

            #t_sph5 = time.time() - t0
            #pl_sph5f.h5f.attrs['t'] = t_sph5
            #pl_sph5f.sph5_create_done()
            #if IsShowSummaryFinished:
            #    pl_sph5f.show_summary_info()
            print('pl spbin file create finished: data shape: %s'%(str(file_data.shape)) )
