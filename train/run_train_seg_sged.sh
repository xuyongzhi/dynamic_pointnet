#!/bin/bash

#***********
# stan 4096 b32: GPU 4500 MB;  CPU 2G
# scannet 8192 b32: GPU 6000 MB; CPU 3G
#***********

train_script=train_semseg_sorted.py
dataset_name=matterport3d
baselogname=log
maxepoch=60
feed_label_elements="label_category,label_instance"
datafeed_type='Pr_Normed_H5f'
feed_data_elements='xyz_1norm_file,xyz_midnorm_block,color_1norm'
feed_data_elements='xyz,color_1norm'

#batchsize=16
#eval_fnglob_or_rate='17DRP5sb8fy'
#all_fn_globs='all_merged_nf5/stride_0d1_step_0d1_pyramid-1d6_2-512_256_64-128_12_6-0d2_0d6_1d2'


all_fn_globs='/home/y/Research/dynamic_pointnet/data/Matterport3D_H5F/v1/scans/2t7WUuJeko7/stride_0d1_step_0d1_pyramid-1d6_2-512_256_64-128_12_6-0d2_0d6_1d2'
eval_fnglob_or_rate='region0.prh5'
batchsize=6


run_train_multifeed="python $train_script --multip_feed  --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batchsize --dataset_name $dataset_name --log_dir $baselogname --datafeed_type $datafeed_type --eval_fnglob_or_rate $eval_fnglob_or_rate --all_fn_globs $all_fn_globs"
run_train="python $train_script --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batchsize --dataset_name $dataset_name --log_dir $baselogname --datafeed_type $datafeed_type --eval_fnglob_or_rate $eval_fnglob_or_rate --all_fn_globs $all_fn_globs"

#$run_train_multifeed
$run_train
