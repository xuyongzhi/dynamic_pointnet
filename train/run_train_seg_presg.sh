#!/bin/bash

#*********** presampling feed
#matterport3D batch_size=16/24 GPU 9G
#			32        >10G
#***********

train_script=train_semseg_sorted.py
dataset_name=matterport3d
baselogname=log
maxepoch=100
batch_size=10
feed_label_elements="label_category-label_instance"
all_fn_globs='v1/each_hosue/stride_0d1_step_0d1_pl_nh5_1d6_2/1'
eval_fnglob_or_rate='17DRP5sb8fy'
bxmh5_folder_name='stride_0d1_step_0d1_bmap_nh5_25600_1d6_2_fmn6-2048_256_64-192_48_6-0d2_0d6_1d2-0d1_0d4_0d8'
#feed_data_elements='xyz_1norm_file-xyz_midnorm_block'
#feed_data_elements='xyz_1norm_file-color_1norm'
feed_data_elements='xyz-color_1norm'
#feed_data_elements='xyz_midnorm_block-color_1norm'
#feed_data_elements='xyz_1norm_file-xyz_midnorm-color_1norm'
model_flag='3AG'
model_flag='3A'    # batch_size=10 9G

#       *******************************************************************
all_fn_globs='v1/scans/stride_0d1_step_0d1_pl_nh5_1d6_2/'
bxmh5_folder_name='stride_0d1_step_0d1_bmap_nh5_25600_1d6_2_fmn3-512_256_64-128_12_6-0d2_0d6_1d2-0d2_0d6_1d2'
feed_data_elements='xyz_midnorm_block'
batch_size=9
model_flag='3A'
# *****************************************************************************
all_fn_globs='v1/scans/stride_0d1_step_0d1_pl_nh5_0d5_1/'
bxmh5_folder_name='stride_0d1_step_0d1_bmap_nh5_2048_0d5_1_fmn3-512_256-128_12-0d2_0d6-0d2_0d6'
feed_data_elements='xyz_midnorm_block'
batch_size=27
model_flag='1AG'
# *****************************************************************************

run_train="python $train_script --model_flag $model_flag  --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batch_size --dataset_name $dataset_name --log_dir $baselogname  --eval_fnglob_or_rate $eval_fnglob_or_rate --all_fn_globs $all_fn_globs --bxmh5_folder_name $bxmh5_folder_name"

run_train_multifeed="python $train_script   --model_flag $model_flag --multip_feed --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batch_size --dataset_name $dataset_name --log_dir $baselogname  --eval_fnglob_or_rate $eval_fnglob_or_rate --all_fn_globs $all_fn_globs --bxmh5_folder_name $bxmh5_folder_name"


finetune_train="python $train_script --model_flag $model_flag  --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batch_size --dataset_name $dataset_name --log_dir $baselogname  --eval_fnglob_or_rate $eval_fnglob_or_rate --all_fn_globs $all_fn_globs --finetune --model_epoch 29 --bxmh5_folder_name $bxmh5_folder_name"

finetune_train_multifeed="python $train_script  --model_flag $model_flag  --multip_feed --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batch_size --dataset_name $dataset_name --log_dir $baselogname  --eval_fnglob_or_rate $eval_fnglob_or_rate --all_fn_globs $all_fn_globs  --finetune --model_epoch 5 --bxmh5_folder_name $bxmh5_folder_name"

$run_train
#$run_train_multifeed
#$finetune_train
#$finetune_train_multifeed
