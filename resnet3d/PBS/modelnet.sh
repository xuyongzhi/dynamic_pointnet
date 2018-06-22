resnet_size=18
optimizer='adam'

batch_norm_decay=0.7

modelnet()
{
  aug_types=$1
  batch_size=$2
  model_flag=$3
  learning_rate0=$4
  num_gpus=$5
  feed_data=$6
  drop_imo=$7
  python ../modelnet_main.py  --resnet_size $resnet_size --model_flag $model_flag --num_gpus 2 --batch_size $batch_size --feed_data $feed_data --aug_types $aug_types --learning_rate0 $learning_rate0 --optimizer $optimizer --batch_norm_decay $batch_norm_decay --learning_rate0 $learning_rate0 --num_gpus $num_gpus --drop_imo $drop_imo
}

modelnet $1 $2 $3 $4 $5 $6 $7
