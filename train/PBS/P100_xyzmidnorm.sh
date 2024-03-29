#!/bin/bash
#PBS -q gpupascal
#PBS -l walltime=10:00:00
#PBS -l mem=18GB
#PBS -l jobfs=0GB
#PBS -l ngpus=1
#PBS -l ncpus=6
## For licensed software, you have to specify it to get the job running. For unlicensed software, you should also specify it to help us analyse the software usage on our system.
#PBS -l software=tensorflow/1.2.1-cudnn6.0-python2.7

## The job will be executed from current working directory instead of home.
#PBS -l wd 
#PBS -r y
#PBS -M yongzhi.xu@student.unsw.edu.au
#PBS -m abe

module load  tensorflow/1.2.1-cudnn6.0-python2.7
source /home/561/yx2146/scripts/set_env.sh
module list
 
./run_train_seg_xyzmidnorm.sh -> out_run_train_seg_xyzmidnorm.log
