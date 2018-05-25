#!/bin/bash
#PBS -q gpupascal
#PBS -l walltime=40:00:00
#PBS -l mem=20GB
#PBS -l jobfs=0GB
#PBS -l ngpus=2
#PBS -l ncpus=12
## For licensed software, you have to specify it to get the job running. For unlicensed software, you should also specify it to help us analyse the software usage on our system.
#PBS -l software=tensorflow/1.6-cudnn7.1-python2.7

## The job will be executed from current working directory instead of home.
#PBS -l wd 
#PBS -r y
#PBS -M y.xu@unsw.edu.au
#PBS -m abe

module load  tensorflow/1.6-cudnn7.1-python2.7
module list
 
./train_seg_presg_scan_1.sh -> out_1.log
