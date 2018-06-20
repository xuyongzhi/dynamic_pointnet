batch_size=64
model_flag='m'
learning_rate0=0.01
num_gpus=2
feed_data='xyzs'
rAnglesYXZ='360_0_0'

aug_types='none'
aug_types='r-360_0_0'
./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data

