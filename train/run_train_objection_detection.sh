#!/bin/bash


train_script = train_obj_detection.py
dataset_name = KITTI
baselogname = log
maxepoch = 30
batchsize = 20
numpoint = 2**15
feed_label_elements = "label_category"

debug_code = "pudb $train_script --dataset_name $dataset_name --log_dir $baselogname --max_epoch $maxepoch --batch_size $batchsize --num_point $numpoint"

run_code = "ipython $train_script --dataset_name $dataset_name --log_dir $baselogname --max_epoch $maxepoch --batch_size $batchsize --num_point $numpoint"



$debug_code
