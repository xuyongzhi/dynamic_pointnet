#!/bin/bash
#PBS -q gpu
#PBS -l walltime=20:00:00
#PBS -l mem=30GB
#PBS -l jobfs=0GB
#PBS -l ngpus=4
#PBS -l ncpus=12
## For licensed software, you have to specify it to get the job running. For unlicensed software, you should also specify it to help us analyse the software usage on our system.
#PBS -l software=tensorflow/1.8-cudnn7.1-python2.7

## The job will be executed from current working directory instead of home.
#PBS -l wd 
#PBS -r y
#PBS -M y.xu@unsw.edu.au
#PBS -m abe

module load  tensorflow/1.8-cudnn7.1-python2.7
module list
 

aug='all'
batch_size=48
model_flag='V'
learning_rate0=0.001
num_gpus=4

../modelnet.sh   $aug  $batch_size $model_flag $learning_rate0 $num_gpus -> a.log

batch_size=96
../modelnet.sh   $aug  $batch_size $model_flag $learning_rate0 $num_gpus -> a96.log
batch_size=48

aug='r'
../modelnet.sh   $aug  $batch_size $model_flag $learning_rate0 $num_gpus -> r.log

aug='s'
../modelnet.sh   $aug  $batch_size $model_flag $learning_rate0 $num_gpus -> s.log

