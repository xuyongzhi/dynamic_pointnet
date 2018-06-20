resnet_size=34
optimizer='adam'

batch_norm_decay=0.9

modelnet()
{
  aug=$1
  batch_size=$2
  model_flag=$3
  learning_rate0=$4
  num_gpus=$5
  feed_data=$6
  python ../modelnet_main.py  --resnet_size $resnet_size --model_flag $model_flag --num_gpus 2 --batch_size $batch_size --feed_data $feed_data --aug $aug --learning_rate0 $learning_rate0 --optimizer $optimizer --batch_norm_decay $batch_norm_decay --learning_rate0 $learning_rate0 --num_gpus $num_gpus
}

modelnet $1 $2 $3 $4 $5 $6
