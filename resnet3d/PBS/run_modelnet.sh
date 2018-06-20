batch_size=64
model_flag='m'
learning_rate0=0.01
num_gpus=2
feed_data='xyzs'


aug='none'
./modelnet.sh   $aug  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data

