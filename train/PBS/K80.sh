#!/bin/bash
#PBS -q gpu
#PBS -l walltime=10:00:00
#PBS -l mem=30GB
#PBS -l jobfs=0GB
#PBS -l ngpus=4
#PBS -l ncpus=12
## For licensed software, you have to specify it to get the job running. For unlicensed software, you should also specify it to help us analyse the software usage on our system.
#PBS -l software=tensorflow/1.6-cudnn7.1-python2.7

## The job will be executed from current working directory instead of home.
#PBS -l wd 
#PBS -r y
#PBS -M y.xu@unsw.edu.au
#PBS -m abe

module load tensorflow/1.6-cudnn7.1-python2.7
module list
 
feed_data_elements='xyz_midnorm_block-color_1norm' 
bs=40
num_gpus=4
in_cnn_out_kp='NN5'
loss_weight='N'
ShuffleFlag='Y'

./train_seg_presg_scan.sh 5VaG_114 $bs $num_gpus $feed_data_elements $loss_weight $in_cnn_out_kp $ShuffleFlag  -> out_NN5.log
