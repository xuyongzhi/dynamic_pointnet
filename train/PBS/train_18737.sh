multip_feed=0
finetune=0
model_epoch=90
only_evaluate=1
ShuffleFlag='Y'
#ShuffleFlag='N'
baselogname=18737/log-model_1bG-gsbb_3B1-bs25-lr1-ds_50-Sf_Y-xyz_midnorm_block-color_1norm-nxnynz-12800-mat_18737
# *****************************************************************************
all_fn_globs='v1/merged_house/stride_0d1_step_0d1_pl_nh5_1d6_2/'
#bxmh5_folder_name='stride_0d1_step_0d1_bmap_nh5_12800_1d6_2_fmn6-2048_256_64-48_32_16-0d2_0d6_1d2-0d1_0d3_0d6'
bxmh5_folder_name='stride_0d1_step_0d1_bmap_nh5_12800_1d6_2_fmn3-512_64_24-48_16_12-0d2_0d6_1d2-0d2_0d6_1d2'
eval_fnglob_or_rate=0.16
# *****************************************************************************


#feed_data_elements='xyz_midnorm_block-color_1norm' 
#feed_data_elements='xyz_midnorm_block' 
feed_data_elements='xyz_midnorm_block-color_1norm-nxnynz' 

./run_train_seg_presg.sh 1bG 25 0 $feed_data_elements $all_fn_globs $bxmh5_folder_name $eval_fnglob_or_rate  $multip_feed $finetune $model_epoch $only_evaluate $ShuffleFlag $baselogname
#./run_train_seg_presg.sh 1bG 25 0 $feed_data_elements $all_fn_globs $bxmh5_folder_name $eval_fnglob_or_rate  $multip_feed $finetune $model_epoch $only_evaluate $ShuffleFlag $baselogname





