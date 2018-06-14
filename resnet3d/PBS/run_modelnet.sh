batch_norm_decay=0.7
learning_rate0=0.001
aug='none'
batch_size=64
optimizer='adam'

./modelnet.sh $batch_norm_decay  $learning_rate0  $aug  $batch_size $optimizer


optimizer='momentum'
./modelnet.sh $batch_norm_decay  $learning_rate0  $aug  $batch_size $optimizer
optimizer='adam'

batch_size=32
./modelnet.sh $batch_norm_decay  $learning_rate0  $aug  $batch_size $optimizer
batch_size=64

aug='all'
./modelnet.sh $batch_norm_decay  $learning_rate0  $aug  $batch_size $optimizer
aug='none'


batch_norm_decay=0.9
./modelnet.sh $batch_norm_decay  $learning_rate0  $aug  $batch_size $optimizer
batch_norm_decay=0.7

learning_rate0=0.0001
./modelnet.sh $batch_norm_decay  $learning_rate0  $aug  $batch_size $optimizer
learning_rate0=0.001
