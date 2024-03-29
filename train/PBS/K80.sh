#!/bin/bash
#PBS -q gpu
#PBS -l walltime=10:00:00
#PBS -l mem=26GB
#PBS -l jobfs=0GB
#PBS -l ngpus=2
#PBS -l ncpus=6
## For licensed software, you have to specify it to get the job running. For unlicensed software, you should also specify it to help us analyse the software usage on our system.
#PBS -l software=tensorflow/1.2.1-cudnn6.0-python2.7

## The job will be executed from current working directory instead of home.
#PBS -l wd 
#PBS -r y
#PBS -M y.xu@student.unsw.edu.au
#PBS -m abe

module load  tensorflow/1.2.1-cudnn6.0-python2.7
#module load tensorflow/1.3.1-cudnn6.0-python2.7
#source /home/561/yx2146/scripts/set_env.sh
module list
 
./run_train_seg_b32_both.sh > out_sem_seg_b32_both.log
