aug='none'
batch_size=48
model_flag='V'
learning_rate0=0.001

#./modelnet.sh   $aug  $batch_size $model_flag $learning_rate0
#
#
#aug='all'
#./modelnet.sh   $aug  $batch_size $model_flag $learning_rate0
#
aug='r'
./modelnet.sh   $aug  $batch_size $model_flag $learning_rate0
#
#aug='s'
#./modelnet.sh   $aug  $batch_size $model_flag $learning_rate0

aug='f'
#./modelnet.sh   $aug  $batch_size $model_flag $learning_rate0

aug='j'
#./modelnet.sh   $aug  $batch_size $model_flag $learning_rate0



#aug='none'
#
#
#learning_rate0=0.0001
#./modelnet.sh   $aug  $batch_size $model_flag $learning_rate0
#
#learning_rate0=0.01
#./modelnet.sh   $aug  $batch_size $model_flag $learning_rate0
#learning_rate0=0.001
#
#batch_size=16
#./modelnet.sh   $aug  $batch_size $model_flag $learning_rate0
#
#batch_size=96
#./modelnet.sh   $aug  $batch_size $model_flag $learning_rate0
