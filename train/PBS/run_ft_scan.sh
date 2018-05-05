
feed_data_elements='xyz_midnorm_block' 
feed_data_elements='xyz_midnorm_block-color_1norm' 

bs=20
num_gpus=1
in_cnn_out_kp='NN5'
loss_weight='N'
ShuffleFlag='Y'
aug=1

./ft_seg_presg_scan.sh 5VaG_114 $bs $num_gpus $feed_data_elements $loss_weight $in_cnn_out_kp $ShuffleFlag  $aug
