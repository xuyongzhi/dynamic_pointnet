aug='none'
batch_siz=64
feed_data='xyzg-nxnynz'
model_flag='m'
resnet_size=34
learning_rate0=0.001
optimizer='adam'

modelnet()
{
  batch_norm_decay=$1
  learning_rate0=$2
  python ../modelnet_main.py  --resnet_size $resnet_size --model_flag $model_flag --num_gpus 2 --batch_size $batch_siz --feed_data $feed_data --aug $aug --learning_rate0 $learning_rate0 --optimizer $optimizer --batch_norm_decay $batch_norm_decay --learning_rate0 $learning_rate0
}

modelnet $1 $2
