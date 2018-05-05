 # modelf_nein=$1
 # batch_size=$2
 # num_gpus=$3
 # feed_data_elements=$4
 # loss_weight=${5}
 # in_cnn_out_kp=${6}

feed_data_elements='xyz_midnorm_block' 
feed_data_elements='xyz_midnorm_block-color_1norm' 


bs=8
num_gpus=2
in_cnn_out_kp='5N5'
loss_weight='N'
ShuffleFlag='Y'
aug=2

./train_seg_presg_scan.sh 5VaG_114 $bs $num_gpus $feed_data_elements $loss_weight $in_cnn_out_kp $ShuffleFlag $aug  -> out_5N5_aug2.log

#-------------------------------
# 5VaG_114 bs=8  466 9.825 1.4

# 5VaG_114 bs=9 NN5  10.414  
# 5VaG_114 bs=9 5N5  10.38  1.153
# 5VaG_114 bs=8 NNN  7.553  0.944
# 5VaG_114 bs=6 N5N T 7.250
# 5VaG_114 bs=6 5NN  5.669 
# 5VaG_114 bs=6 55N T 7.221  1.20
# 5VaG_114 bs=6 555 T 8.402  1.40
