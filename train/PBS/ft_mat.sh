#feed_data_elements='xyz_midnorm_block-color_1norm' 
#feed_data_elements='xyz_midnorm_block' 
feed_data_elements='xyz_midnorm_block-color_1norm-nxnynz' 
#feed_data_elements='xyz_midnorm_block-nxnynz' 

#./run_train_seg_presg_mat.sh 2aG_144 30 0 $feed_data_elements E
#./run_train_seg_presg_mat.sh 3aG_444 45 0 $feed_data_elements E
#./run_train_seg_presg_mat.sh 4bG_114 20 1 $feed_data_elements E
#./run_train_seg_presg_mat.sh 4bG_144 18 1 $feed_data_elements E
#./run_train_seg_presg_mat.sh 4bG_444 15 1 $feed_data_elements E
./ft_train_seg_presg_mat.sh 4bG_111 20 0 $feed_data_elements E

