batch_size=32
model_flag='m'
learning_rate0=0.001
num_gpus=2
feed_data='xyzsg-nxnynz'
drop_imo='0_0_5'
aug_types='rsfj-360_0_0'

./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo
