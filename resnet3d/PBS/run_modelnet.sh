batch_size=64
model_flag='m'
learning_rate0=0.01
num_gpus=2
feed_data='xyzs'
drop_imo='0_0_5'
aug_types='none'

./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo

batch_size=32
./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo
batch_size=64

drop_imo='0_0_7'
./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo

drop_imo='0_0_3'
./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo
