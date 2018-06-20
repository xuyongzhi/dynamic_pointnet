batch_size=64
model_flag='m'
learning_rate0=0.01
num_gpus=2
feed_data='xyzs'

#aug_types='r-360_0_0'
#./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data

aug_types='r-0_360_0'
./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data

