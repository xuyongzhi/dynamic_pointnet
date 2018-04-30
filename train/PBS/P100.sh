#!/bin/bash
#PBS -q gpupascal
#PBS -l walltime=15:00:00
#PBS -l mem=30GB
#PBS -l jobfs=0GB
#PBS -l ngpus=3
#PBS -l ncpus=18
## For licensed software, you have to specify it to get the job running. For unlicensed software, you should also specify it to help us analyse the software usage on our system.
#PBS -l software=tensorflow/1.6-cudnn7.1-python2.7

## The job will be executed from current working directory instead of home.
#PBS -l wd 
#PBS -r y
#PBS -M y.xu@unsw.edu.au
#PBS -m abe

module load  tensorflow/1.6-cudnn7.1-python2.7
module list
 
feed_data_elements='xyz_midnorm_block-color_1norm' 
bs=30
num_gpus=3
in_cnn_out_kp='587'  # 14.2G
loss_weight='N'

./train_seg_presg_scan.sh 5VaG_114 $bs $num_gpus $feed_data_elements $loss_weight $in_cnn_out_kp  -> out_587.log

in_cnn_out_kp='5N5'
./train_seg_presg_scan.sh 5VaG_114 $bs $num_gpus $feed_data_elements $loss_weight $in_cnn_out_kp -> out_5N5.log

