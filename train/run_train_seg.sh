#!/bin/bash

#*********** presampling feed
#matterport3D batchsize=16/24 GPU 9G
#			32        >10G
#***********

train_script=train_semseg_sorted.py
dataset_name=matterport3d
baselogname=log
maxepoch=10
batchsize=24
feed_label_elements="label_category,label_instance"
datafeed_type='Pr_Normed_H5f'
all_fn_globs='stride_0d1_step_0d1_pyramid-1d6_2-512_256_64-128_12_6-0d2_0d6_1d2/house_groups'
eval_fnglob_or_rate='WYY-X7H-XcA-YFu-YVU-YmJ-Z6M-ZMo-aay-ac2-b8c-cV4-dhj-e9z-fzy'
#feed_data_elements='xyz_1norm_file,xyz_midnorm_block'
feed_data_elements='xyz_1norm_file,color_1norm'


run_train="python $train_script  --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batchsize --dataset_name $dataset_name --log_dir $baselogname --datafeed_type $datafeed_type --eval_fnglob_or_rate $eval_fnglob_or_rate --all_fn_globs $all_fn_globs"

run_train_multifeed="python $train_script  --multip_feed --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batchsize --dataset_name $dataset_name --log_dir $baselogname --datafeed_type $datafeed_type --eval_fnglob_or_rate $eval_fnglob_or_rate --all_fn_globs $all_fn_globs"

#$run_train
$run_train_multifeed
