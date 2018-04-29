feed_data_elements='xyz_midnorm_block' 
feed_data_elements='xyz_midnorm_block-color_1norm' 


 # modelf_nein=$1
 # batch_size=$2
 # num_gpus=$3
 # feed_data_elements=$4
 # loss_weight=${5}
 # in_cnn_out_kp=${6}

 bs=10
 num_gpus=2
 in_cnn_out_kp=555
 loss_weight='N'

./train_seg_presg_scan.sh 5VaG_114 $bs $num_gpus $feed_data_elements $loss_weight $in_cnn_out_kp 


 in_cnn_out_kp=595
./train_seg_presg_scan.sh 5VaG_114 $bs $num_gpus $feed_data_elements $loss_weight $in_cnn_out_kp 

 in_cnn_out_kp=955
./train_seg_presg_scan.sh 5VaG_114 $bs $num_gpus $feed_data_elements $loss_weight $in_cnn_out_kp 


#-------------------------------
# 5VaG_114 bs=8 9.173G

# 5VaG_114 bs=6 NNN 5.671G  1.133
# 5VaG_114 bs=6 NN5 6.797G  1.133
# 5VaG_114 bs=6 5N5 6.802G
# 5VaG_114 bs=4 N55 N 7.993G 1.998
# 5VaG_114 bs=6 N55 T 8.393G 1.399
