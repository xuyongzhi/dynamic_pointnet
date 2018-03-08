#!/bin/bash


train_script=train_obj_detection.py
debug_script=train_obj_detection_debug.py
dataset_name=rawh5_kitti
baselogname=log
maxepoch=40
batchsize=5
numpoint=32768
feed_label_elements="label_category"
model_epoch=1



# d_code="python $debug_script --dataset_name $dataset_name --log_dir $baselogname --max_epoch $maxepoch --batch_size $batchsize --num_point $numpoint"

train_code="python $train_script --dataset_name $dataset_name --log_dir $baselogname --max_epoch $maxepoch --batch_size $batchsize --num_point $numpoint"

train_code2="python $train_script --dataset_name $dataset_name --log_dir $baselogname --max_epoch $maxepoch --batch_size $batchsize --num_point $numpoint --finetune --model_epoch $model_epoch"

$train_code
#$train_code2
