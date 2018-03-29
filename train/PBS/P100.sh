#!/bin/bash
#PBS -q gpupascal
#PBS -l walltime=24:00:00
#PBS -l mem=26GB
#PBS -l jobfs=0GB
#PBS -l ngpus=2
#PBS -l ncpus=12
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

./run_train_seg_presg_scan.sh 4bG_111 22 0 $feed_data_elements 'E' -> out_scan111.log &
./run_train_seg_presg_scan.sh 4bG_144 18 1 $feed_data_elements 'E' -> out_scan144.log
