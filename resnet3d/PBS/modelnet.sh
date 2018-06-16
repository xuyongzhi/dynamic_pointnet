feed_data='xyzg-nxnynz'
resnet_size=34
optimizer='adam'

batch_norm_decay=0.9
#learning_rate0=0.001
#aug='none'
#batch_size=48

modelnet()
{
  aug=$1
  batch_size=$2
  model_flag=$3
  learning_rate0=$4
  python ../modelnet_main.py  --resnet_size $resnet_size --model_flag $model_flag --num_gpus 2 --batch_size $batch_size --feed_data $feed_data --aug $aug --learning_rate0 $learning_rate0 --optimizer $optimizer --batch_norm_decay $batch_norm_decay --learning_rate0 $learning_rate0
}

modelnet $1 $2 $3 $4
