#!/bin/bash
#PBS -q gpupascal
#PBS -l walltime=12:00:00
#PBS -l mem=16GB
#PBS -l jobfs=0GB
#PBS -l ngpus=1
#PBS -l ncpus=6
## For licensed software, you have to specify it to get the job running. For unlicensed software, you should also specify it to help us analyse the software usage on our system.
#PBS -l software=tensorflow/1.4-cudnn6.0-python2.7

## The job will be executed from current working directory instead of home.
#PBS -l wd 
#PBS -r y
#PBS -M yongzhi.xu@student.unsw.edu.au
#PBS -m abe

module load  tensorflow/1.4-cudnn6.0-python2.7
module list
 
#feed_data_elements='xyz_midnorm_block-color_1norm' 
feed_data_elements='xyz_midnorm_block-color_1norm-nxnynz' 
#./run_train_seg_presg.sh 1aG 30 $feed_data_elements 0 -> out_1aG30n.log 
./run_train_seg_presg.sh 1bG 27 $feed_data_elements 0 -> out_1bG27n.log 
