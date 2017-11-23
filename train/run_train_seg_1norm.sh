

python train_semseg.py  --gpu 0    --log_dir log6_4096_B8_tmp_xyz1norm  --test_area 6  --max_epoch 10 --batch_size 8 --num_point 4096  --dataset_name scannet --channel_elementes  xyz_1norm


#python train_semseg.py  --gpu 0    --log_dir log6_4096_B8_tmp_xyzmidnorm  --test_area 6  --max_epoch 10 --batch_size 8 --num_point 4096  --dataset_name stanford_indoor --channel_elementes  xyz_midnorm

#python train_semseg.py  --gpu 0    --log_dir log6_4096_B32_tmp_xyzmidnorm  --test_area 6  --max_epoch 10 --batch_size 32 --num_point 4096  --dataset_name stanford_indoor --channel_elementes  xyz_midnorm
#python train_semseg.py  --gpu 0    --log_dir log6_4096_B32_tmp_xyz1norm_xyzmidnorm  --test_area 6  --max_epoch 10 --batch_size 32 --num_point 4096  --dataset_name stanford_indoor --channel_elementes  xyz_1norm,xyz_midnorm

#python train_semseg.py  --gpu 0    --log_dir log6_4096_A_tmp  --test_area 6  --max_epoch 10 --batch_size 32 --num_point 4096  --max_test_file_num 2 -- dataset_name stanford_indoor



