#!/bin/bash

train_script=train_semseg.py


logdir="lograijin3-B32-xyz_1norm-4096-stanford_indoor"

stan_b32="python $train_script --only_evaluate --test_are 6 --batch_size 32 --dataset_name stanford_indoor --log_dir $logdir"

#**********************************************   Tmp: all data, one epoch



#**********************************************   Tmp: small data, auto_break

stan_b4_f10="python $train_script --only_evaluate --test_are 6 --batch_size 2 --dataset_name stanford_indoor --log_dir log1_4096_xyz1norm_epoch47_testacc0.65_trainacc0.91 --max_test_file_num 10"



#./parallel_commands "$stan_b32_xyz1norm_e1" "$scannet_b32_xyz1norm_e1"
#./parallel_commands "$stan_b8_xyz1norm_e3_f30" "$scannet_b8_xyz1norm_e3_f30"

$stan_b32


#***********
# stan 4096 b32: GPU 4500 MB;  CPU 2G
# scannet 8192 b32: GPU 6000 MB; CPU 3G
