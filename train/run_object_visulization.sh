#!/bin/bash

run_script=object_detection_visulization.py
dataset_name=rawh5_kitti
baselogname=log
maxepoch=30
model_epoch=15
batchsize=1
numpoint=32768
num_batches=2



run_code="pudb $run_script --dataset_name $dataset_name --log_dir $baselogname --num_batches $num_batches --max_epoch $maxepoch --batch_size $batchsize --num_point $numpoint --model_epoch $model_epoch --only_evaluate"

$run_code

