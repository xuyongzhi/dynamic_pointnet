#!/bin/bash

#*********** presampling feed
#matterport3D batch_size=16/24 GPU 9G
#			32        >10G
#***********

train_script=../train_semseg_sorted_multi_gpus.py
dataset_name=scannet
maxepoch=501
learning_rate=0.002
decay_epoch_step=50
feed_label_elements="label_category"

multip_feed=1
finetune=0
model_epoch=170
only_evaluate=0
ShuffleFlag='Y'
#ShuffleFlag='N'
baselogname=log

# *****************************************************************************
all_fn_globs='Merged_sph5/gs-6_-10/'
bxmh5_folder_name='Merged_bxmh5/320000_gs-6_-10_fmn4-8000_4800_320_56-100_20_40_32-0d1_0d4_1_2d4-0d1_0d2_0d6_1d2-3B3'

all_fn_globs='Merged_sph5/128000_gs-6_-10/'
bxmh5_folder_name='Merged_bxmh5/128000_gs-6_-10_fmn4-8000_4800_320_64-24_20_40_32-0d1_0d4_1_2d4-0d1_0d2_0d6_1d2-3B4'

eval_fnglob_or_rate=0
#eval_fnglob_or_rate='train_300'
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