feed_data_elements='xyz_midnorm_block' 
feed_data_elements='xyz_midnorm_block-color_1norm' 


./ft_seg_presg_scan.sh 4aG_114 18 2 $feed_data_elements E 1.3   # 0.17G per batch -> 60 for 1080TI, 90 for P100
