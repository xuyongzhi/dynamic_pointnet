#!/bin/bash

#*********** presampling feed
#matterport3D batch_size=16/24 GPU 9G
#			32        >10G
#***********

train_script=../train_semseg_sorted.py
dataset_name=matterport3d
maxepoch=201
learning_rate=0.005
decay_epoch_step=60
feed_label_elements="label_category-label_instance"

multip_feed=1
finetune=1
model_epoch=200
only_evaluate=0
ShuffleFlag='Y'
#ShuffleFlag='N'
baselogname=log1
baselogname='log1-model_4bG-Elw-gsbb_3A1-bs20-lr5-ds_60-Sf_Y-xyz_midnorm_block-color_1norm-nxnynz-12800-mat_60'
# *****************************************************************************
#all_fn_globs='v1/merged_house/stride_0d1_step_0d1_pl_nh5_1d6_2/17D_1LX_1pX_29h_2az'
all_fn_globs='v1/merged_house/stride_0d1_step_0d1_pl_nh5_1d6_2/'
#bxmh5_folder_name='stride_0d1_step_0d1_bmap_nh5_12800_1d6_2_fmn6-2048_256_64-48_32_16-0d2_0d6_1d2-0d1_0d3_0d6'
bxmh5_folder_name='stride_0d1_step_0d1_bmap_nh5_12800_1d6_2_fmn3-512_64_24-48_16_12-0d2_0d6_1d2-0d2_0d6_1d2'
eval_fnglob_or_rate=0.16
# *****************************************************************************
#all_fn_globs='v1/merged_house/stride_0d1_step_0d1_pl_nh5_1d6_2/17D_1LX_1pX_29h_2az,v1/each_house/stride_0d1_step_0d1_pl_nh5_1d6_2/zsNo4HB9uLZ'
all_fn_globs='v1/each_house/stride_0d1_step_0d1_pl_nh5-1d6_2/17DRP5sb8fy'
bxmh5_folder_name='stride_0d1_step_0d1_bxmh5-12800_1d6_2_fmn4-480_80_24-80_20_10-0d2_0d6_1d2-0d2_0d6_1d2-3A1'
eval_fnglob_or_rate=0
# *****************************************************************************

run_train()
{ 
  model_flag=$1
  batch_size=$2
  gpu=$3
  feed_data_elements=$4

  loss_weight=${5}
  python $train_script --model_flag $model_flag  --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batch_size --dataset_name $dataset_name --log_dir $baselogname  --eval_fnglob_or_rate $eval_fnglob_or_rate --all_fn_globs $all_fn_globs --bxmh5_folder_name $bxmh5_folder_name --learning_rate $learning_rate --multip_feed $multip_feed --finetune $finetune --model_epoch $model_epoch --gpu $gpu --only_evaluate $only_evaluate --decay_epoch_step $decay_epoch_step --ShuffleFlag $ShuffleFlag --loss_weight $loss_weight
}

run_train $1 $2 $3 $4  $5
