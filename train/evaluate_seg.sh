#!/bin/bash

train_script=train_semseg.py

#python $train_script --test_area 6 --max_epoch 50 --batch_size 32 --num_point 4096  --dataset_name stanford_indoor --channel_elementes xyz_1norm --learning_rate 0.001 --log_dir log


#python $train_script --test_area 6 --max_epoch 50 --batch_size 32 --num_point 8192  --dataset_name scannet --channel_elementes xyz_1norm --learning_rate 0.001 --log_dir log





#**********************************************   Tmp: all data, one epoch
stan_b32="python $train_script --only_evaluate --test_are 6 --batch_size 32 --dataset_name stanford_indoor --log_dir log1_4096_xyz1norm_epoch47_testacc0.65_trainacc0.91"

stan_b64_xyz1norm_e1="python $train_script --test_area 6 --max_epoch 1 --batch_size 32 --num_point 4096  --dataset_name stanford_indoor --channel_elementes xyz_1norm --learning_rate 0.001 --log_dir log_tmp"
stan_b128_xyz1norm_e1="python $train_script --test_area 6 --max_epoch 1 --batch_size 128 --num_point 4096  --dataset_name stanford_indoor --channel_elementes xyz_1norm --learning_rate 0.001 --log_dir log_tmp"


scannet_b32_xyz1norm_e1="python $train_script --max_epoch 1 --batch_size 32 --num_point 8192  --dataset_name scannet --channel_elementes xyz_1norm --learning_rate 0.001 --log_dir log_tmp --auto_break"
#**********************************************   Tmp: small data, auto_break

stan_b4_f10="python $train_script --only_evaluate --test_are 6 --batch_size 2 --dataset_name stanford_indoor --log_dir log1_4096_xyz1norm_epoch47_testacc0.65_trainacc0.91 --max_test_file_num 10"

stan_b8_xyz1norm_e3_f30="python $train_script --test_area 6 --max_epoch 3 --batch_size 8 --num_point 4096  --dataset_name stanford_indoor --channel_elementes xyz_1norm --learning_rate 0.001 --log_dir log_tmp --max_test_file_num 30"
scannet_b8_xyz1norm_e3_f30="python $train_script --max_epoch 3 --batch_size 8 --num_point 8192  --dataset_name scannet --channel_elementes xyz_1norm --learning_rate 0.001 --log_dir log_tmp --max_test_file_num 30"


#./parallel_commands "$stan_b32_xyz1norm_e1" "$scannet_b32_xyz1norm_e1"
#./parallel_commands "$stan_b8_xyz1norm_e3_f30" "$scannet_b8_xyz1norm_e3_f30"

$stan_b32


#***********
# stan 4096 b32: GPU 4500 MB;  CPU 2G
# scannet 8192 b32: GPU 6000 MB; CPU 3G
