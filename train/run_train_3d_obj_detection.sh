#!/bin/bash

#*********** presampling feed
#matterport3D batch_size=16/24 GPU 9G
#			32        >10G
#***********

train_script=./train_3d_obj_detection.py
dataset_name=KITTI
maxepoch=301
learning_rate=0.002
decay_epoch_step=50
feed_label_elements='bounding_box'

multip_feed=1
finetune=0
model_epoch=100
only_evaluate=0
ShuffleFlag='Y'
#ShuffleFlag='N'
baselogname='log'

# *****************************************************************************
all_fn_globs='MergedicData/ORG_sph5/4000_mgs10_gs5_10-mbf-neg/'
bxmh5_folder_name='ORG_bxmh5/4000_mgs10_gs5_10-mbf-neg_fmn-10-10-10_mvp2-1600_1800_800_1-16_16_24_32-0d4_1d2_2d4-0d2_0d3_0d4-pd3-3D1_benz'

eval_fnglob_or_rate=0
# *****************************************************************************

modelf_nein='3aG_114'
batch_size=4
#num_gpus=1
feed_data_elements='xyzrs'
#feed_data_elements='xyz_1norm'
loss_weight='E'
group_pos='center'
gpu=1



train_script="python $train_script --modelf_nein $modelf_nein --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batch_size --dataset_name $dataset_name --log_dir $baselogname  --eval_fnglob_or_rate $eval_fnglob_or_rate --all_fn_globs $all_fn_globs --bxmh5_folder_name $bxmh5_folder_name --group_pos $group_pos --learning_rate $learning_rate --model_epoch $model_epoch --loss_weight $loss_weight --substract_center --gpu=$gpu" 

$train_script
