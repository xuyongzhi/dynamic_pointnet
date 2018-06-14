aug='all'
aug='none'
batch_siz=64
feed_data='xyzg-nxnynz'
model_flag='V'
python ../modelnet_main.py --residual  --resnet_size 34 --model_flag $model_flag --num_gpus 2 --batch_size $batch_siz --feed_data $feed_data --aug $aug

python ../modelnet_main.py  --resnet_size 34 --model_flag $model_flag --num_gpus 2 --batch_size $batch_siz --feed_data $feed_data --aug $aug


model_flag=''
#python ../modelnet_main.py --residual 'True' --resnet_size 34 --model_flag $model_flag --num_gpus 2 --batch_size $batch_siz --feed_data $feed_data --aug $aug


