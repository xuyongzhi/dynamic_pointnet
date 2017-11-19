
python train_semseg.py  --gpu 0    --log_dir log6_4096_B_tmp  --test_area 6  --max_epoch 10 --batch_size 8 --num_point 4096  --dataset_name stanford_indoor
#python train_semseg.py  --gpu 0    --log_dir log6_4096_A_tmp  --test_area 6  --max_epoch 10 --batch_size 32 --num_point 4096  --max_test_file_num 2 -- dataset_name stanford_indoor



#python train_semseg.py  --gpu 0    --log_dir log1_4096_WC_tmp  --test_area 1  --max_epoch 3 --batch_size 2 --num_point 4096 --max_test_file_num 2 --dataset_name scannet
