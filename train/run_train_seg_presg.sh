#!/bin/bash

#*********** presampling feed
#matterport3D batch_size=16/24 GPU 9G
#			32        >10G
#***********

train_script=train_semseg_sorted.py
dataset_name=matterport3d
baselogname=log
maxepoch=100
learning_rate=0.01
feed_label_elements="label_category-label_instance"
all_fn_globs='v1/each_hosue/stride_0d1_step_0d1_pl_nh5_1d6_2/1'
eval_fnglob_or_rate=0.3
#       *******************************************************************
#all_fn_globs='v1/scans/stride_0d1_step_0d1_pl_nh5_1d6_2/'
#bxmh5_folder_name='stride_0d1_step_0d1_bmap_nh5_25600_1d6_2_fmn3-512_256_64-128_12_6-0d2_0d6_1d2-0d2_0d6_1d2'
# *****************************************************************************
all_fn_globs='v1/scans/stride_0d1_step_0d1_pl_nh5_0d5_1/'
bxmh5_folder_name='stride_0d1_step_0d1_bmap_nh5_2048_0d5_1_fmn1-160_32-32_12-0d2_0d6-0d2_0d6'
# *****************************************************************************
batch_size=81
model_flag='1AG'

run_train()
{ 
  model_flag=$1
  batch_size=$2
  feed_data_elements=$3
  multip_feed=$4
  finetune=$finetune
  python $train_script --model_flag $model_flag  --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batch_size --dataset_name $dataset_name --log_dir $baselogname  --eval_fnglob_or_rate $eval_fnglob_or_rate --all_fn_globs $all_fn_globs --bxmh5_folder_name $bxmh5_folder_name --learning_rate $learning_rate --multip_feed $multip_feed --finetune $finetune
}

multip_feed=0
finetune=0

#feed_data_elements='xyz_1norm_file-xyz_midnorm_block'
#feed_data_elements='xyz_1norm_file-color_1norm'
#feed_data_elements='xyz'
#feed_data_elements='xyz-color_1norm'
feed_data_elements='xyz_midnorm_block-color_1norm'
#feed_data_elements='xyz_1norm_file-xyz_midnorm-color_1norm'


#run_train $model_flag $batch_size $feed_data_elements $multip_feed $finetune

run_train '1AG' 27 $feed_data_elements $multip_feed $finetune
run_train '1AG' 81 $feed_data_elements $multip_feed $finetune
run_train '3AG' 81 $feed_data_elements $multip_feed $finetune
run_train '2A' 81 $feed_data_elements $multip_feed $finetune
