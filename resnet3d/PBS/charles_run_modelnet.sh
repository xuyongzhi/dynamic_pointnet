batch_size=20
model_flag='m'
learning_rate0=0.001
num_gpus=1
feed_data='xyzrsg-nxnynz'
drop_imo='0_0_5'
aug_types='rpsfj-360_0_0'
#aug_types='N'

./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo

