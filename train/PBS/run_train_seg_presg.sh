#!/bin/bash

#*********** presampling feed
#matterport3D batch_size=16/24 GPU 9G
#			32        >10G
#***********

train_script=../train_semseg_sorted.py
dataset_name=matterport3d
baselogname=log
maxepoch=501
learning_rate=0.005
feed_label_elements="label_category-label_instance"

# *****************************************************************************
all_fn_globs='v1/merged_house/stride_0d1_step_0d1_pl_nh5_1d6_2/'
bxmh5_folder_name='stride_0d1_step_0d1_bmap_nh5_12800_1d6_2_fmn6-2048_256_64-48_32_16-0d2_0d6_1d2-0d1_0d3_0d6'
eval_fnglob_or_rate='JeF_Jmb_PX4_Pm6_PuK'
# *****************************************************************************
#all_fn_globs='v1/each_house/stride_0d1_step_0d1_pl_nh5_1d6_2/'
#bxmh5_folder_name='stride_0d1_step_0d1_bmap_nh5_12800_1d6_2_fmn6-2048_256_64-48_32_16-0d2_0d6_1d2-0d1_0d3_0d6'
#eval_fnglob_or_rate=0.5
# *****************************************************************************
#all_fn_globs='v1/small_test/stride_0d1_step_0d1_pl_nh5_1d6_2/'
#bxmh5_folder_name='stride_0d1_step_0d1_bmap_nh5_12800_1d6_2_fmn6-2048_256_64-32_32_16-0d2_0d6_1d2-0d1_0d3_0d6'
#eval_fnglob_or_rate='region1'
# *****************************************************************************

run_train()
{ 
  model_flag=$1
  batch_size=$2
  feed_data_elements=$3
  gpu=$4
  multip_feed=$5
  finetune=$6
  model_epoch=$7
  python $train_script --model_flag $model_flag  --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batch_size --dataset_name $dataset_name --log_dir $baselogname  --eval_fnglob_or_rate $eval_fnglob_or_rate --all_fn_globs $all_fn_globs --bxmh5_folder_name $bxmh5_folder_name --learning_rate $learning_rate --multip_feed $multip_feed --finetune $finetune --model_epoch $model_epoch --gpu $gpu
}

multip_feed=0
finetune=0
model_epoch=40

#feed_data_elements='xyz_1norm_file-xyz_midnorm_block'
#feed_data_elements='xyz_1norm_file-color_1norm'
#feed_data_elements='xyz'
#feed_data_elements='xyz-color_1norm'
#feed_data_elements='xyz_midnorm_block-color_1norm'
#feed_data_elements='xyz_1norm_file-xyz_midnorm-color_1norm'


#run_train $model_flag $batch_size $feed_data_elements $multip_feed $finetune

#run_train '1AG' 32  $feed_data_elements 0 $multip_feed $finetune $model_epoch
run_train $1 $2 $3 $4  $multip_feed $finetune $model_epoch
