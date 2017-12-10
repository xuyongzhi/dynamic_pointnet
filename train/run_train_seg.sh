#!/bin/bash

#***********
# stan 4096 b32: GPU 4500 MB;  CPU 2G
# scannet 8192 b32: GPU 6000 MB; CPU 3G
#***********

train_script=train_semseg_sorted.py
#dataset_name=stanford_indoor
dataset_name=scannet
baselogname=log

b32_xyz1norm="python $train_script --max_epoch 30 --batch_size 32 --num_point 8192  --dataset_name $dataset_name --feed_elements xyz_1norm --learning_rate 0.001 --log_dir $baselogname"
b32_xyzmidnorm="python $train_script --max_epoch 30 --batch_size 32 --num_point 8192  --dataset_name $dataset_name --feed_elements xyz_midnorm --learning_rate 0.001 --log_dir $baselogname"
b32_xyz="python $train_script --max_epoch 30 --batch_size 32 --num_point 8192  --dataset_name $dataset_name --feed_elements xyz --learning_rate 0.001 --log_dir $baselogname"


#$b32_xyz
#$b32_xyz1norm


./parallel_commands "$b32_xyz1norm" "$b32_xyzmidnorm" "$b32_xyz"

