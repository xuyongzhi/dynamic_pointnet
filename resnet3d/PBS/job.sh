#!/bin/bash
#PBS -q gpu
#PBS -l walltime=20:00:00
#PBS -l mem=30GB
#PBS -l jobfs=0GB
#PBS -l ngpus=2
#PBS -l ncpus=6
## For licensed software, you have to specify it to get the job running. For unlicensed software, you should also specify it to help us analyse the software usage on our system.
#PBS -l software=tensorflow/1.8-cudnn7.1-python2.7

## The job will be executed from current working directory instead of home.
#PBS -l wd 
#PBS -r y
##PBS -M y.xu@unsw.edu.au
##PBS -m abe

module load  tensorflow/1.8-cudnn7.1-python2.7
module list
 

batch_size=32
model_flag='m'
learning_rate0=0.001
num_gpus=2
feed_data='xyzsg-nxnynz'
drop_imo='0_0_5'
aug_types='rsfj-360_0_0'

./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo -> out32.log

aug_types='N'
./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo -> out32.log
aug_types='rsfj-360_0_0'

learning_rate0=0.01
./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo -> out32.log
learning_rate0=0.001

aug_types='sfj'
./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo -> out32.log
