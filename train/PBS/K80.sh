#!/bin/bash
#PBS -q gpu
#PBS -l walltime=10:00:00
#PBS -l mem=20GB
#PBS -l jobfs=0GB
#PBS -l ngpus=2
#PBS -l ncpus=6
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
bs=20
num_gpus=2
in_cnn_out_kp='5N5'
loss_weight='N'
ShuffleFlag='Y'
aug=2

./train_seg_presg_scan.sh 5VaG_114 $bs $num_gpus $feed_data_elements $loss_weight $in_cnn_out_kp $ShuffleFlag $aug  -> out_5N5_aug2.log
