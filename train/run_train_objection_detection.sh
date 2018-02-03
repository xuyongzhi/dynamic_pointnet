#!/bin/bash


train_script=train_obj_detection.py
debug_script=train_obj_detection_debug.py
dataset_name=rawh5_kitti
baselogname=log
maxepoch=30
batchsize=2
numpoint=32768
feed_label_elements="label_category"

d_code="python $debug_script --dataset_name $dataset_name --log_dir $baselogname --max_epoch $maxepoch --batch_size $batchsize --num_point $numpoint"

run_code="python $train_script --dataset_name $dataset_name --log_dir $baselogname --max_epoch $maxepoch --batch_size $batchsize --num_point $numpoint"



$d_code
