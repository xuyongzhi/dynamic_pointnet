#!/bin/bash
#PBS -q gpu
#PBS -l walltime=20:00:00
#PBS -l mem=26GB
#PBS -l jobfs=0GB
#PBS -l ngpus=2
#PBS -l ncpus=6
## For licensed software, you have to specify it to get the job running. For unlicensed software, you should also specify it to help us analyse the software usage on our system.
#PBS -l software=tensorflow/1.8-cudnn7.1-python2.7

## The job will be executed from current working directory instead of home.
#PBS -l wd 
#PBS -r y
#PBS -M y.xu@unsw.edu.au
#PBS -m abe

module load  tensorflow/1.8-cudnn7.1-python2.7
module list
 

batch_size=48
model_flag='m'
learning_rate0=0.001


aug='f'
./modelnet.sh   $aug  $batch_size $model_flag $learning_rate0 -> f.log

aug='j'
./modelnet.sh   $aug  $batch_size $model_flag $learning_rate0 -> j.log

aug='all'
./modelnet.sh   $aug  $batch_size $model_flag $learning_rate0 -> a.log
