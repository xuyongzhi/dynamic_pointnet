feed_data='xyzg-nxnynz'
model_flag='m'
resnet_size=34

batch_norm_decay=0.7
learning_rate0=0.001
aug='none'
batch_size=64

modelnet()
{
  batch_norm_decay=$1
  learning_rate0=$2
  aug=$3
  batch_size=$4
  optimizer=$5
  python ../modelnet_main.py  --resnet_size $resnet_size --model_flag $model_flag --num_gpus 2 --batch_size $batch_size --feed_data $feed_data --aug $aug --learning_rate0 $learning_rate0 --optimizer $optimizer --batch_norm_decay $batch_norm_decay --learning_rate0 $learning_rate0
}

modelnet $1 $2 $3 $4 $5
