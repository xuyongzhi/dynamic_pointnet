aug='all'
batch_siz=32 # 9.2G
feed_data='xyzg-nxnynz'
#python ../modelnet_main.py --num_gpus 2 --batch_size $batch_siz --feed_data $feed_data --aug $aug

batch_siz=64
#python ../modelnet_main.py --num_gpus 2 --batch_size $batch_siz --feed_data $feed_data --aug $aug

#feed_data='xyzg'
python ../modelnet_main.py --resnet_size 34 --model_flag '3Vm' --num_gpus 2 --batch_size $batch_siz --feed_data $feed_data --aug $aug

aug='none'
#python ../modelnet_main.py --num_gpus 2 --batch_size $batch_siz --feed_data $feed_data --aug $aug

