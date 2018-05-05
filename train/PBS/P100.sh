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
num_gpus=2
loss_weight='N'
ShuffleFlag='Y'
bs=26
in_cnn_out_kp='NN5'
aug=1

./train_seg_presg_scan.sh 5VaG_114 $bs $num_gpus $feed_data_elements $loss_weight $in_cnn_out_kp $ShuffleFlag $aug  -> out_NN5.log


