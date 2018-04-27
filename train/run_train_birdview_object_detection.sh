#!/bin/bash

#*********** presampling feed
#matterport3D batch_size=16/24 GPU 9G
#			32        >10G
#***********

train_script=./train_birdview_obj_detection.py
dataset_name=Voxel
maxepoch=301
learning_rate=0.002
decay_epoch_step=50
feed_label_elements='label_category'

multip_feed=1
finetune=0
model_epoch=100
only_evaluate=0
ShuffleFlag='Y'
#ShuffleFlag='N'
baselogname='log'

# *****************************************************************************
all_fn_globs='Merged_sph5/32768_gs-100_-100/'
bxmh5_folder_name='Merged_bxmh5/32768_gs-100_-100_fmn-10-10-10-12800_4800_2400-16_8_8-0d4_0d8_1d8-0d2_0d4_0d4-pd1-3D3'

eval_fnglob_or_rate=0.0
# *****************************************************************************

modelf_nein='3a_11'
batch_size=8
#num_gpus=1
feed_data_elements='xyz'
loss_weight='E'
group_pos='center'


train_script="python $train_script --modelf_nein $modelf_nein  --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batch_size --dataset_name $dataset_name --log_dir $baselogname  --eval_fnglob_or_rate $eval_fnglob_or_rate --all_fn_globs $all_fn_globs --bxmh5_folder_name $bxmh5_folder_name --group_pos $group_pos --learning_rate $learning_rate --model_epoch $model_epoch --loss_weight $loss_weight" 

$train_script
