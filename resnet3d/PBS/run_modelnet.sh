batch_norm_decay=0.997
learning_rate0=0.001
./modelnet.sh $batch_norm_decay $learning_rate0

batch_norm_decay=0.5
./modelnet.sh $batch_norm_decay $learning_rate0
