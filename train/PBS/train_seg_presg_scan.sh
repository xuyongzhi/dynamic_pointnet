#!/bin/bash

#*********** presampling feed
#matterport3D batch_size=16/24 GPU 9G
#			32        >10G
#***********

train_script=../train_semseg_sorted_multi_gpus.py
#dataset_name=SCANNET
dataset_name=MODELNET40
maxepoch=51
learning_rate=0.002
decay_epoch_step=50
feed_label_elements="label_category"

multip_feed=1
finetune=0
model_epoch=170
only_evaluate=0
baselogname=log
# *****************************************************************************
finetune=0
model_epoch=40
baselogname='log-5VaG_114-Nlw5N5-gsbb_4B1-bs25-lr2-ds_40-Sf_Y-xyz_midnorm_block-color_1norm-30000-SCA_12319-augRef-bd7'
baselogname='log'
# *****************************************************************************
all_fn_globs='Merged_sph5/90000_gs-3d6_-6d3/'
bxmh5_folder_name='Merged_bxmh5/90000_gs-3d6_-6d3_fmn1444-6400_2400_320_32-32_16_32_48-0d1_0d3_0d9_2d7-0d1_0d2_0d6_1d8-pd3-mbf-4A1'
eval_fnglob_or_rate='test'

all_fn_globs='Merged_sph5/30000_gs-2d4_-3d4/'
bxmh5_folder_name='Merged_bxmh5/30000_gs-2d4_-3d4_fmn1444-2048_1024_128_24-48_32_48_27-0d1_0d4_1_2d2-0d1_0d2_0d6_1d2-pd3-mbf-4B1'

#all_fn_globs='Merged_sph5/30000_gs-2d4_-3d4/,Merged_sph5/30000_gs-2d4_-3d4-dec5/'
#bxmh5_folder_name='Merged_bxmh5/30000_gs-2d4_-3d4_fmn1444-2048_1024_128_24-48_32_48_27-0d1_0d4_1_2d2-0d1_0d2_0d6_1d2-pd3-mbf-4B1,Merged_bxmh5/30000_gs-2d4_-3d4_fmn1444-2048_1024_128_24-48_32_48_27-0d1_0d4_1_2d2-0d1_0d2_0d6_1d2-pd3-mbf-4B1-dec5'

all_fn_globs='Merged_sph5/10000_gs2_2d3/'
bxmh5_folder_name='Merged_bxmh5/10000_gs2_2d3_fmn1444-2560_1024_80_16-24_32_48_32-0d0_0d2_0d5_1d1-0d0_0d1_0d3_0d6-pd3-mbf-4M2'

all_fn_globs='Merged_sph5/10000_gs2_2/'
bxmh5_folder_name='Merged_bxmh5/10000_gs2_2_fmn1444-2048_960_64_12-24_32_48_24-0d0_0d2_0d5_1d1-0d0_0d1_0d3_0d6-pd3-mbf-4M1'

all_fn_globs='Merged_sph5/1024_gs2_2d3/'
bxmh5_folder_name='Merged_bxmh5/1024_gs2_2d3_fmn1444-640_160-32_32-0d2_0d4-0d1_0d2-pd3-2M'
# *****************************************************************************

run_train()
{ 
  modelf_nein=$1
  batch_size=$2
  num_gpus=$3
  feed_data_elements=$4
  loss_weight=$5
  in_cnn_out_kp=$6
  ShuffleFlag=$7
  aug=$8
  python $train_script --modelf_nein $modelf_nein  --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batch_size --dataset_name $dataset_name --log_dir $baselogname  --eval_fnglob_or_rate $eval_fnglob_or_rate --all_fn_globs $all_fn_globs --bxmh5_folder_name $bxmh5_folder_name --learning_rate $learning_rate --multip_feed $multip_feed --finetune $finetune --model_epoch $model_epoch --num_gpus $num_gpus --only_evaluate $only_evaluate --decay_epoch_step $decay_epoch_step --ShuffleFlag $ShuffleFlag --loss_weight $loss_weight --in_cnn_out_kp $in_cnn_out_kp --aug $aug --start_gi 0
}

run_train $1 $2 $3 $4  $5 $6 $7 $8
