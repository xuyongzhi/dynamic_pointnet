#!/bin/bash

#***********
# stan 4096 b32: GPU 4500 MB;  CPU 2G
# scannet 8192 b32: GPU 6000 MB; CPU 3G
#***********

train_script=train_semseg_sorted.py
#dataset_name=stanford_indoor
dataset_name=matterport3d
baselogname=log
maxepoch=30
batchsize=48
numpoint=8192
#feed_label_elements="label_category,label_instance"
feed_label_elements="label_category"

b48_xyz1norm="python $train_script --feed_data_elements xyz_1norm --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batchsize --num_point $numpoint  --dataset_name $dataset_name --log_dir $baselogname"
b48_xyzmidnorm="python $train_script --feed_data_elements xyz_midnorm_block  --feed_label_elements $feed_label_elements --max_epoch $maxepoch --batch_size $batchsize --num_point $numpoint  --dataset_name $dataset_name --log_dir $baselogname"
b48_xyz="python $train_script --feed_data_elements xyz  --feed_label_elements $feed_label_elements --max_epoch $maxepoch --batch_size $batchsize --num_point $numpoint  --dataset_name $dataset_name --log_dir $baselogname"


#./parallel_commands "$b48_xyz1norm" "$b48_xyzmidnorm" "$b48_xyz"
#$b48_xyzmidnorm

b48_xyz1norm_FT="python $train_script --feed_data_elements xyz_1norm  --feed_label_elements $feed_label_elements --max_epoch $maxepoch --batch_size $batchsize --num_point $numpoint  --dataset_name $dataset_name --log_dir $baselogname --finetune --model_epoch 5"

#./parallel_commands "$b48_xyz1norm"
#$b48_xyzmidnorm


#----------------  fast code test with small data
matterport_fnglob="v1*"
scanet_small_fnglob="v1/scans/17DRP5sb8fy/stride-2-step-4_8192_normed/"
small_test_xyz1norm="python $train_script --feed_data_elements xyz_midnorm_block  --feed_label_elements $feed_label_elements --all_fn_globs $matterport_fnglob --eval_fnglob_or_rate 0.4   --max_epoch 3 --batch_size 8  --num_point $numpoint  --dataset_name $dataset_name --log_dir small_data_test_log --auto_break --datafeed_type Normed_H5f"


pyramid_matterport_fnglob="v1/scans/17DRP5sb8fy/stride_0d1_step_0d1_pyramid-1_2-256_64_16-0d2_0d4_0d8"
pyramid_small_test_xyz1norm="python  $train_script --feed_data_elements xyz_midnorm_block  --feed_label_elements $feed_label_elements --all_fn_globs $pyramid_matterport_fnglob --eval_fnglob_or_rate 0.4   --max_epoch 3 --batch_size 2  --dataset_name $dataset_name --log_dir small_data_test_log --auto_break --datafeed_type Pr_Normed_H5f"
$pyramid_small_test_xyz1norm
