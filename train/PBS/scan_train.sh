feed_data_elements='xyz_midnorm_block' 


#./run_train_seg_presg_scan.sh 4aG_114 55 1 $feed_data_elements CN 0.3   # 0.17G per batch -> 60 for 1080TI, 90 for P100
./run_train_seg_presg_scan.sh 4aG_114 55 1 $feed_data_elements E 1.3   # 0.17G per batch -> 60 for 1080TI, 90 for P100
#./run_train_seg_presg_scan.sh 4aG_114 55 0 $feed_data_elements CN 1.3   # 0.17G per batch -> 60 for 1080TI, 90 for P100


#./run_train_seg_presg_scan.sh 4aG_144 50 0 $feed_data_elements CN 1.3  # 0.2G per batch -> 50 for 1080TI, 80 for P100
#./run_train_seg_presg_scan.sh 4aG_144 50 0 $feed_data_elements CN 0.3  # 0.2G per batch -> 50 for 1080TI, 80 for P100


#./run_train_seg_presg_scan.sh 4aG_114 58 1 $feed_data_elements E 1.3  # 0.17G per batch -> 60 for 1080TI, 90 for P100
#./run_train_seg_presg_scan.sh 4aG_114 58 1 $feed_data_elements N 1.3   # 0.17G per batch -> 60 for 1080TI, 90 for P100

./run_train_seg_presg_scan.sh 4aG_144 40 0 $feed_data_elements E  1.3  # 0.2G per batch -> 50 for 1080TI, 80 for P100
#./run_train_seg_presg_scan.sh 4aG_144 50 0 $feed_data_elements N  1.3  # 0.2G per batch -> 50 for 1080TI, 80 for P100



#./run_train_seg_presg_scan.sh 4bG_144 18 1 $feed_data_elements 'E'

#./run_train_seg_presg_scan.sh 3aG_144 48 1 $feed_data_elements 'E'
#./run_train_seg_presg_scan.sh 4bG_111 22 1 $feed_data_elements 'E'
#./run_train_seg_presg_scan.sh 4bG_144 18 1 $feed_data_elements 'E'

