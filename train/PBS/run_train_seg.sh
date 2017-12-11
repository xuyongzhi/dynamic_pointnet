#!/bin/bash

#***********
# stan 4096 b32: GPU 4500 MB;  CPU 2G
# scannet 8192 b32: GPU 6000 MB; CPU 3G
#***********

train_script=train_semseg_sorted.py
#dataset_name=stanford_indoor
dataset_name=scannet
baselogname=log
maxepoch=30
batchsize=48
numpoint=8192

b48_xyz1norm="python $train_script --feed_elements xyz_1norm --max_epoch $maxepoch --batch_size $batchsize --num_point $numpoint  --dataset_name $dataset_name --log_dir $baselogname"
b48_xyzmidnorm="python $train_script --feed_elements xyz_midnorm --max_epoch $maxepoch --batch_size $batchsize --num_point $numpoint  --dataset_name $dataset_name --log_dir $baselogname"
b48_xyz="python $train_script --feed_elements xyz --max_epoch $maxepoch --batch_size $batchsize --num_point $numpoint  --dataset_name $dataset_name --log_dir $baselogname"


#./parallel_commands "$b48_xyz1norm"

