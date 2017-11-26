#!/bin/bash

train_script=train_semseg.py

#python $train_script --test_area 6 --max_epoch 50 --batch_size 32 --num_point 4096  --dataset_name stanford_indoor --channel_elementes xyz_1norm --learning_rate 0.001 --log_dir log


#python $train_script --test_area 6 --max_epoch 50 --batch_size 32 --num_point 8192  --dataset_name scannet --channel_elementes xyz_1norm --learning_rate 0.001 --log_dir log





#**********************************************   Tmp: all data, one epoch
stan_b32_xyz1norm_e1="python $train_script --test_area 6 --max_epoch 1 --batch_size 32 --num_point 4096  --dataset_name stanford_indoor --channel_elementes xyz_1norm --learning_rate 0.001 --log_dir log_tmp"
stan_b64_xyz1norm_e1="python $train_script --test_area 6 --max_epoch 1 --batch_size 32 --num_point 4096  --dataset_name stanford_indoor --channel_elementes xyz_1norm --learning_rate 0.001 --log_dir log_tmp"
stan_b128_xyz1norm_e1="python $train_script --test_area 6 --max_epoch 1 --batch_size 128 --num_point 4096  --dataset_name stanford_indoor --channel_elementes xyz_1norm --learning_rate 0.001 --log_dir log_tmp"


#**********************************************   Tmp: small data, auto_break


stan_b1_e1_f10="python $train_script --test_area 6 --max_epoch 2 --batch_size 1 --num_point 4096  --dataset_name stanford_indoor --channel_elementes xyz_1norm --learning_rate 0.001 --log_dir log_tmp --max_test_file_num 10"
stan_b32_e1__f10="python $train_script --test_area 6 --max_epoch 2 --batch_size 32 --num_point 4096  --dataset_name stanford_indoor --channel_elementes xyz_1norm --learning_rate 0.001 --log_dir log_tmp --max_test_file_num 10"
#python $train_script --max_epoch 2 --batch_size 2 --num_point 8192  --dataset_name scannet --channel_elementes xyz_1norm --learning_rate 0.001 --log_dir log_tmp --max_test_file_num 10 --auto_break


./parallel_commands "$stan_b32_xyz1norm_e1" "$stan_b64_xyz1norm_e1" "$stan_b128_xyz1norm_e1"
