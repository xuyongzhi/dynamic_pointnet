
train_model='train_sorted_sem_seg.py'
echo $train_model

#python  $train_model   --gpu 0    --log_dir log1_4096_WC_tmp  --eval_area 1  --max_epoch 4 --batch_size 1 --num_point 4096    --data_elements xyz_1norm       --all_fn_glob stanford_indoor3d_globalnormedh5_stride_0.5_step_1_4096/*WC_1*  --train_data_rate 1 --eval_data_rate 1



python $train_model  --gpu 0    --log_dir log6_4096_xyz1norm  --eval_area 6 --max_epoch 10 --batch_size 32 --num_point 4096  --data_elements xyz_1norm    --all_fn_glob stanford_indoor3d_globalnormedh5_stride_0.5_step_1_4096/*






