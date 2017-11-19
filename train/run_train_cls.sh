

cls_model='pointnet2_cls_msg'
cls_model='pointnet2_cls_ssg'


python train_cls.py --gpu 0 --model pointnet2_cls_msg --log_dir log --num_point 512 --max_epoch 2 --batch_size 1 --learning_rate 0.001 --momentum 0.9 --optimizer adam --decay_step 200000 --decay_rate 0.7


#python train_cls.py --gpu 0 --model pointnet2_cls_msg --log_dir log --num_point 1024 --max_epoch 250 --batch_size 32 --learning_rate 0.001 --momentum 0.9 --optimizer adam --decay_step 200000 --decay_rate 0.7
