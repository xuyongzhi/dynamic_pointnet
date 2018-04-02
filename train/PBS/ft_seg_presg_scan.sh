#!/bin/bash

#*********** presampling feed
#matterport3D batch_size=16/24 GPU 9G
#			32        >10G
#***********

train_script=../train_semseg_sorted.py
dataset_name=scannet
maxepoch=301
learning_rate=0.002
decay_epoch_step=30
feed_label_elements="label_category"

multip_feed=0
finetune=0
model_epoch=300
only_evaluate=1
ShuffleFlag='Y'
#ShuffleFlag='N'
baselogname=log
baselogname='scan_12887/log-model_3aG_144-Elw-gsbb_3A1-bs48-lr2-ds_30-Sf_Y-xyz_midnorm_block-12800-sca_12887'
# *****************************************************************************
all_fn_globs='each_house/stride_0d1_step_0d1_pl_nh5-1d6_2/'
bxmh5_folder_name='stride_0d1_step_0d1_bxmh5-12800_1d6_2_fmn4-480_80_24-80_20_10-0d2_0d6_1d2-0d2_0d6_1d2-3A1'
eval_fnglob_or_rate='test'
#eval_fnglob_or_rate='train_300'
# *****************************************************************************

run_train()
{ 
  modelf_nein=$1
  batch_size=$2
  gpu=$3
  feed_data_elements=$4
  loss_weight=${5}
  inkp_min=${6}
  python $train_script --modelf_nein $modelf_nein  --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batch_size --dataset_name $dataset_name --log_dir $baselogname  --eval_fnglob_or_rate $eval_fnglob_or_rate --all_fn_globs $all_fn_globs --bxmh5_folder_name $bxmh5_folder_name --learning_rate $learning_rate --multip_feed $multip_feed --finetune $finetune --model_epoch $model_epoch --gpu $gpu --only_evaluate $only_evaluate --decay_epoch_step $decay_epoch_step --ShuffleFlag $ShuffleFlag --loss_weight $loss_weight --inkp_min $inkp_min
}

run_train $1 $2 $3 $4  $5 ${6}
