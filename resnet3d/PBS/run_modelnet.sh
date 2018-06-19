aug='none'
batch_size=48
model_flag='m'
learning_rate0=0.001
num_gpus=2


aug='all'
./modelnet.sh   $aug  $batch_size $model_flag $learning_rate0 $num_gpus

aug='r'
./modelnet.sh   $aug  $batch_size $model_flag $learning_rate0 $num_gpus

aug='s'
./modelnet.sh   $aug  $batch_size $model_flag $learning_rate0 $num_gpus

aug='f'
#./modelnet.sh   $aug  $batch_size $model_flag $learning_rate0 $num_gpus

aug='j'
#./modelnet.sh   $aug  $batch_size $model_flag $learning_rate0 $num_gpus

