#!/bin/bash
#PBS -q gpupascal
#PBS -l walltime=36:00:00
#PBS -l mem=46GB
#PBS -l jobfs=0GB
#PBS -l ngpus=4
#PBS -l ncpus=24
## For licensed software, you have to specify it to get the job running. For unlicensed software, you should also specify it to help us analyse the software usage on our system.
#PBS -l software=tensorflow/1.6-cudnn7.1-python2.7

## The job will be executed from current working directory instead of home.
#PBS -l wd 
#PBS -r y
#PBS -M yongzhi.xu@student.unsw.edu.au
#PBS -m abe

module load  tensorflow/1.6-cudnn7.1-python2.7
module list
 
#feed_data_elements='xyz_midnorm_block-color_1norm' 
#feed_data_elements='xyz_midnorm_block-color_1norm-nxnynz' 
feed_data_elements='xyz_midnorm_block' 

./run_train_seg_presg_mat.sh 4aG_114 90 0 $feed_data_elements E 0.3   -> out_scan114_E.log &
./run_train_seg_presg_mat.sh 4aG_114 90 0 $feed_data_elements CN 0.3   -> out_scan114_CN.log &
./run_train_seg_presg_mat.sh 4aG_114 90 0 $feed_data_elements CN 1.3   -> out_scan114_nodrop.log &
./run_train_seg_presg_mat.sh 4aG_144 80 0 $feed_data_elements CN 0.3   -> out_scan144.log 

