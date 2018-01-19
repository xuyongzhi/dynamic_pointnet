#!/bin/bash

#***********
# stan 4096 b32: GPU 4500 MB;  CPU 2G
# scannet 8192 b32: GPU 6000 MB; CPU 3G
#***********

train_script=train_semseg_sorted.py
dataset_name=matterport3d
baselogname=log
maxepoch=60
batchsize=16
feed_label_elements="label_category,label_instance"
datafeed_type='Pr_Normed_H5f'
eval_fnglob_or_rate='17DRP5sb8fy'
feed_data_elements='xyz_1norm_file,xyz_midnorm_block,color_1norm'
feed_data_elements='xyz,color_1norm'

run_train_multifeed="python $train_script --multip_feed  --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batchsize --dataset_name $dataset_name --log_dir $baselogname --datafeed_type $datafeed_type --eval_fnglob_or_rate $eval_fnglob_or_rate"
run_train="python $train_script --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batchsize --dataset_name $dataset_name --log_dir $baselogname --datafeed_type $datafeed_type --eval_fnglob_or_rate $eval_fnglob_or_rate"

#$run_train_multifeed
$run_train
