dataset_name='scannet'
feed_label_elements="label_category-label_instance"
feed_label_elements="label_category"
multip_feed=1
finetune=0
model_epoch=30
only_evaluate=0
loss_weight='CN'
ShuffleFlag='Y'
baselogname=log
# *****************************************************************************
all_fn_globs='each_house/stride_0d1_step_0d1_pl_nh5_1d6_2/'
bxmh5_folder_name='stride_0d1_step_0d1_bmap_nh5_12800_1d6_2_fmn3-256_48_16-56_8_8-0d2_0d6_1d2-0d2_0d6_1d2'
eval_fnglob_or_rate='test'
# *****************************************************************************


#feed_data_elements='xyz_midnorm_block-color_1norm' 
feed_data_elements='xyz_midnorm_block' 
#feed_data_elements='xyz_midnorm_block-color_1norm-nxnynz' 
#feed_data_elements='xyz_midnorm_block-nxnynz' 

./run_train_seg_presg_scannet.sh 1bG 25 0 $feed_data_elements $all_fn_globs $bxmh5_folder_name $eval_fnglob_or_rate  $multip_feed $finetune $model_epoch $only_evaluate $ShuffleFlag $baselogname $loss_weight

