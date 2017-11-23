train_script=train_semseg.py

#python $train_script --test_area 6 --max_epoch 50 --batch_size 32 --num_point 4096  --dataset_name stanford_indoor --channel_elementes xyz_1norm --learning_rate 0.001 --log_dir log


#python $train_script --test_area 6 --max_epoch 50 --batch_size 32 --num_point 8192  --dataset_name scannet --channel_elementes xyz_1norm --learning_rate 0.001 --log_dir log






#**********************************************   Tmp


python $train_script --test_area 6 --max_epoch 2 --batch_size 8 --num_point 4096  --dataset_name stanford_indoor --channel_elementes xyz_1norm --learning_rate 0.001 --log_dir log_tmp --max_test_file_num 11 --auto_break
#python $train_script --max_epoch 2 --batch_size 2 --num_point 8192  --dataset_name scannet --channel_elementes xyz_1norm --learning_rate 0.001 --log_dir log_tmp --max_test_file_num 10 --auto_break
