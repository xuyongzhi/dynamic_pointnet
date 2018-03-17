#!/bin/bash

#*********** presampling feed
#matterport3D batch_size=16/24 GPU 9G
#			32        >10G
#***********

train_script=../train_semseg_sorted.py
dataset_name='scannet'
maxepoch=301
learning_rate=0.001
decay_epoch_step=40
feed_label_elements="label_category"


run_train()
{ 
  model_flag=$1
  batch_size=$2
  gpu=$3
  feed_data_elements=$4
  all_fn_globs=$5
  bxmh5_folder_name=$6
  eval_fnglob_or_rate=$7
  multip_feed=$8
  finetune=$9
  model_epoch=${10}
  only_evaluate=${11}
  ShuffleFlag=${12}
  baselogname=${13}
  loss_weight=${14}
  python $train_script --model_flag $model_flag  --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batch_size --dataset_name $dataset_name --log_dir $baselogname  --eval_fnglob_or_rate $eval_fnglob_or_rate --all_fn_globs $all_fn_globs --bxmh5_folder_name $bxmh5_folder_name --learning_rate $learning_rate --multip_feed $multip_feed --finetune $finetune --model_epoch $model_epoch --gpu $gpu --only_evaluate $only_evaluate --decay_epoch_step $decay_epoch_step --ShuffleFlag $ShuffleFlag --loss_weight $loss_weight
}




run_train $1 $2 $3 $4  $5 $6 $7 $8 $9 ${10} ${11} ${12} ${13} ${14}
