#!/bin/bash

#*********** presampling feed
#matterport3D batch_size=16/24 GPU 9G
#			32        >10G
#***********

train_script=../train_semseg_sorted_multi_gpus.py
dataset_name=scannet
maxepoch=201
learning_rate=0.001
decay_epoch_step=40
feed_label_elements="label_category"

multip_feed=1
finetune=1
model_epoch=100
only_evaluate=0
ShuffleFlag='Y'
#ShuffleFlag='N'
baselogname=log
#baselogname='3540_traingood_testbad/log-model_4aG_114-Nlw-gsbb_3C2-bs18-lr2-ds_50-Sf_Y-xyz_midnorm_block-color_1norm-60000-sca_3540'
baselogname='log-model_4bG_114-Nlw-gsbb_3C1-bs16-lr2-ds_50-Sf_Y-xyz_midnorm_block-color_1norm-90000-sca_2051'
# *****************************************************************************
all_fn_globs='Merged_sph5/gs-6_-10/'
bxmh5_folder_name='Merged_bxmh5/320000_gs-6_-10_fmn4-8000_4800_320_56-100_20_40_32-0d1_0d4_1_2d4-0d1_0d2_0d6_1d2-3B3'

all_fn_globs='Merged_sph5/128000_gs-6_-10/'
bxmh5_folder_name='Merged_bxmh5/128000_gs-6_-10_fmn4-8000_4800_320_64-24_20_40_32-0d1_0d4_1_2d4-0d1_0d2_0d6_1d2-3B4'

all_fn_globs='Merged_sph5/60000_gs-3_-4d8/'
bxmh5_folder_name='Merged_bxmh5/60000_gs-3_-4d8_fmn6-1600_480_48-80_16_32-0d2_0d6_1d8-0d2_0d4_1d2-3C2'

all_fn_globs='Merged_sph5/90000_gs-4_-6d3/'   #  4aG_114 - bs:5*2
bxmh5_folder_name='Merged_bxmh5/90000_gs-4_-6d3_fmn6-6400_2400_300_30-32_10_24_32-0d1_0d3_0d9_2d7-0d1_0d2_0d6_1d8-3C1'

#eval_fnglob_or_rate='trainval_scene0507_00_to_0706_00-298'
eval_fnglob_or_rate='test'
# *****************************************************************************

run_train()
{ 
  modelf_nein=$1
  batch_size=$2
  num_gpus=$3
  feed_data_elements=$4
  loss_weight=${5}
  inkp_min=${6}
  python $train_script --modelf_nein $modelf_nein  --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batch_size --dataset_name $dataset_name --log_dir $baselogname  --eval_fnglob_or_rate $eval_fnglob_or_rate --all_fn_globs $all_fn_globs --bxmh5_folder_name $bxmh5_folder_name --learning_rate $learning_rate --multip_feed $multip_feed --finetune $finetune --model_epoch $model_epoch --num_gpus $num_gpus --only_evaluate $only_evaluate --decay_epoch_step $decay_epoch_step --ShuffleFlag $ShuffleFlag --loss_weight $loss_weight --inkp_min $inkp_min
}

run_train $1 $2 $3 $4  $5 $6