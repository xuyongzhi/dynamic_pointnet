#!/bin/bash

#*********** presampling feed
#matterport3D batch_size=16/24 GPU 9G
#			32        >10G
#***********

train_script=../train_semseg_sorted_multi_gpus.py
dataset_name=scannet
maxepoch=501
learning_rate=0.002
decay_epoch_step=40
feed_label_elements="label_category"

multip_feed=1
finetune=0
model_epoch=170
only_evaluate=0
baselogname=log

# *****************************************************************************
all_fn_globs='Merged_sph5/90000_gs-3d6_-6d3/'
bxmh5_folder_name='Merged_bxmh5/90000_gs-3d6_-6d3_fmn1444-6400_2400_320_32-32_16_32_48-0d1_0d3_0d9_2d7-0d1_0d2_0d6_1d8-pd3-mbf-4A1'
eval_fnglob_or_rate='test'
# *****************************************************************************

run_train()
{ 
  modelf_nein=$1
  batch_size=$2
  num_gpus=$3
  feed_data_elements=$4
  loss_weight=$5
  in_cnn_out_kp=$6
  ShuffleFlag=$7
  aug=$8
  python $train_script --modelf_nein $modelf_nein  --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batch_size --dataset_name $dataset_name --log_dir $baselogname  --eval_fnglob_or_rate $eval_fnglob_or_rate --all_fn_globs $all_fn_globs --bxmh5_folder_name $bxmh5_folder_name --learning_rate $learning_rate --multip_feed $multip_feed --finetune $finetune --model_epoch $model_epoch --num_gpus $num_gpus --only_evaluate $only_evaluate --decay_epoch_step $decay_epoch_step --ShuffleFlag $ShuffleFlag --loss_weight $loss_weight --in_cnn_out_kp $in_cnn_out_kp --aug $aug
}

run_train $1 $2 $3 $4  $5 $6 $7 $8
