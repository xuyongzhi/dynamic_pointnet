batch_size=32
model_flag='m'
learning_rate0=0.001
num_gpus=2
feed_data='xyzrsg-nxnynz'
drop_imo='0_0_5'
aug_types='rpsfj-360_0_0'
num_filters0=32
optimizer='adam'
aug_types='psfj'
#aug_types='N'


./modelnet.sh   $aug_types  $batch_size $model_flag $learning_rate0 $num_gpus $feed_data $drop_imo $num_filters0 $optimizer

