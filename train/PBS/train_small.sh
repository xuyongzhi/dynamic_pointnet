multip_feed=1
finetune=0
model_epoch=301
only_evaluate=0
ShuffleFlag='Y'
#ShuffleFlag='N'
baselogname=logsmall
# *****************************************************************************
#all_fn_globs='v1/merged_house/stride_0d1_step_0d1_pl_nh5_1d6_2/17D_1LX_1pX_29h_2az'
all_fn_globs='v1/merged_house/stride_0d1_step_0d1_pl_nh5_1d6_2/'
#bxmh5_folder_name='stride_0d1_step_0d1_bmap_nh5_12800_1d6_2_fmn6-2048_256_64-48_32_16-0d2_0d6_1d2-0d1_0d3_0d6'
bxmh5_folder_name='stride_0d1_step_0d1_bmap_nh5_12800_1d6_2_fmn3-512_64_24-48_16_12-0d2_0d6_1d2-0d2_0d6_1d2'
eval_fnglob_or_rate=0.16
# *****************************************************************************


#feed_data_elements='xyz_midnorm_block-color_1norm' 
#feed_data_elements='xyz_midnorm_block' 
feed_data_elements='xyz_midnorm_block-color_1norm-nxnynz' 
#feed_data_elements='xyz_midnorm_block-nxnynz' 

#./run_train_seg_presg.sh 1aG 30 0 $feed_data_elements $all_fn_globs $bxmh5_folder_name $eval_fnglob_or_rate  $multip_feed $finetune $model_epoch $only_evaluate $ShuffleFlag
./run_train_seg_presg.sh 4bG 25 0 $feed_data_elements $all_fn_globs $bxmh5_folder_name $eval_fnglob_or_rate  $multip_feed $finetune $model_epoch $only_evaluate $ShuffleFlag $baselogname





