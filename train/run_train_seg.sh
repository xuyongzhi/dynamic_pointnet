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


<<<<<<< HEAD
#./parallel_commands "$b48_xyz1norm" "$b48_xyzmidnorm" "$b48_xyz"
$b48_xyzmidnorm
=======
>>>>>>> fcc2c6c1b49e9496277b125cdbfbe1cdd8feebfd

b48_xyz1norm_FT="python $train_script --feed_elements xyz_1norm --max_epoch $maxepoch --batch_size $batchsize --num_point $numpoint  --dataset_name $dataset_name --log_dir $baselogname --finetune --model_epoch 5"

#./parallel_commands "$b48_xyz1norm"
#$b48_xyzmidnorm
$b48_xyz1norm_FT


#----------------  fast code test with small data
small_test_xyz1norm="python $train_script --feed_elements xyz_1norm --all_fn_globs stride_1_step_2_test_small_8192_normed/ --eval_fnglob_or_rate 0.4   --max_epoch 3 --batch_size 4  --num_point $numpoint  --dataset_name $dataset_name --log_dir small_data_test_log"
#$small_test_xyz1norm
