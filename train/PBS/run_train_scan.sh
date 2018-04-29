 # modelf_nein=$1
 # batch_size=$2
 # num_gpus=$3
 # feed_data_elements=$4
 # loss_weight=${5}
 # in_cnn_out_kp=${6}

feed_data_elements='xyz_midnorm_block' 
feed_data_elements='xyz_midnorm_block-color_1norm' 

bs=14
num_gpus=2
in_cnn_out_kp='466'
loss_weight='N'

./train_seg_presg_scan.sh 5VaG_114 $bs $num_gpus $feed_data_elements $loss_weight $in_cnn_out_kp 


#in_cnn_out_kp='4N6'
#./train_seg_presg_scan.sh 5VaG_114 $bs $num_gpus $feed_data_elements $loss_weight $in_cnn_out_kp 
#
#in_cnn_out_kp='N66'
#./train_seg_presg_scan.sh 5VaG_114 $bs $num_gpus $feed_data_elements $loss_weight $in_cnn_out_kp 


#-------------------------------
# 5VaG_114 bs=8  466 9.825 1.4

# 5VaG_114 bs=6 NNN  5.672  0.9453
# 5VaG_114 bs=8 NNN  7.553  0.944
# 5VaG_114 bs=6 NN5  6.802  1.134
# 5VaG_114 bs=6 N5N T 7.250
# 5VaG_114 bs=6 N5N N 10.757
# 5VaG_114 bs=6 5NN  5.669 
# 5VaG_114 bs=6 55N T 7.221  1.20
# 5VaG_114 bs=6 555 T 8.402  1.40
