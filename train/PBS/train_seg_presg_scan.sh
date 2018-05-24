#matterport3D batch_size=16/24 GPU 9G
#*********** presampling feed
#			32        >10G
#***********

train_script=../train_semseg_sorted_multi_gpus.py
#dataset_name=SCANNET
dataset_name=MODELNET40
maxepoch=101
learning_rate=0.001
decay_epoch_step=30
feed_label_elements="label_category"
multip_feed=1

finetune=0
model_epoch=170
only_evaluate=0
baselogname='log'
# *****************************************************************************
#finetune=1
#model_epoch=100
#baselogname='4m/log-4m-ElwNN5-xyz_rsg-mean-gsbb_3M1-bs16-lr1-ds_30-Sf_Y-xyzrsg-nxnynz-4096-MOD_9843-augIn-bd5'
# *****************************************************************************
all_fn_globs='Merged_sph5/90000_gs-3d6_-6d3/'
bxmh5_folder_name='Merged_bxmh5/90000_gs-3d6_-6d3_fmn1444-6400_2400_320_32-32_16_32_48-0d1_0d3_0d9_2d7-0d1_0d2_0d6_1d8-pd3-mbf-4A1'
eval_fnglob_or_rate='test'

all_fn_globs='Merged_sph5/30000_gs-2d4_-3d4/'
bxmh5_folder_name='Merged_bxmh5/30000_gs-2d4_-3d4_fmn1444-2048_1024_128_24-48_32_48_27-0d1_0d4_1_2d2-0d1_0d2_0d6_1d2-pd3-mbf-4B1'

#all_fn_globs='Merged_sph5/30000_gs-2d4_-3d4/,Merged_sph5/30000_gs-2d4_-3d4-dec5/'
#bxmh5_folder_name='Merged_bxmh5/30000_gs-2d4_-3d4_fmn1444-2048_1024_128_24-48_32_48_27-0d1_0d4_1_2d2-0d1_0d2_0d6_1d2-pd3-mbf-4B1,Merged_bxmh5/30000_gs-2d4_-3d4_fmn1444-2048_1024_128_24-48_32_48_27-0d1_0d4_1_2d2-0d1_0d2_0d6_1d2-pd3-mbf-4B1-dec5'

all_fn_globs='Merged_sph5/10000_gs3_3d5/'
bxmh5_folder_name='Merged_bxmh5/10000_gs3_3d5_fmn1444_mvp1-2560_1024_80_16_1-24_32_48_27_48-0d0_0d2_0d5_1d1-0d0_0d1_0d3_0d6-pd3-mbf-neg-4M1'

#all_fn_globs='Merged_sph5/4096_mgs1_gs2_2d2/'
#bxmh5_folder_name='Merged_bxmh5/4096_mgs1_gs2_2d2_fmn1444_mvp1-3200_1024_48_1-18_24_56_56-0d1_0d2_0d6-0d0_0d1_0d4-pd3-mbf-neg-3M1'
#
#all_fn_globs='Merged_sph5/4096_mgs1_gs2_2d2_nmbf/'
#bxmh5_folder_name='Merged_bxmh5/4096_mgs1_gs2_2d2_fmn1444_mvp1-3200_1024_48_1-18_24_56_56-0d1_0d2_0d6-0d0_0d1_0d4-pd3-neg-3M1'

#all_fn_globs='Merged_sph5/4096_mgs1_gs2_2/'
#bxmh5_folder_name='Merged_bxmh5/4096_mgs1_gs2_2_fmn14_mvp1-1024_240_1-48_27_160-0d2_0d4-0d1_0d2-pd3-mbf-neg-2M2p'

# *****************************************************************************

run_train()
{ 
  modelf_nein=$1
  batch_size=$2
  num_gpus=$3
  feed_data_elements=$4
  group_pos=$5
  loss_weight=$6
  in_cnn_out_kp=$7
  ShuffleFlag=$8
  aug=$9
  start_gi=${10}
  python $train_script --modelf_nein $modelf_nein  --feed_data_elements $feed_data_elements --feed_label_elements $feed_label_elements  --max_epoch $maxepoch --batch_size $batch_size --dataset_name $dataset_name --log_dir $baselogname  --eval_fnglob_or_rate $eval_fnglob_or_rate --all_fn_globs $all_fn_globs --bxmh5_folder_name $bxmh5_folder_name --learning_rate $learning_rate --multip_feed $multip_feed --finetune $finetune --model_epoch $model_epoch --num_gpus $num_gpus --only_evaluate $only_evaluate --decay_epoch_step $decay_epoch_step --ShuffleFlag $ShuffleFlag --loss_weight $loss_weight --in_cnn_out_kp $in_cnn_out_kp --aug $aug --start_gi $start_gi --group_pos $group_pos
}

#run_train $1 $2 $3 $4  $5 $6 $7 $8 $9 ${10}


#-------------------------------------------------------------------------------------------
#feed_data_elements='xyzg' 
#feed_data_elements='xyzrsg-color_1norm' 
feed_data_elements='xyzg-nxnynz' 

num_gpus=1
start_gi=0
in_cnn_out_kp='NN5'
loss_weight='E'
ShuffleFlag='Y'
group_pos='mean'
aug=1

bs=20
run_train 5m $bs $num_gpus $feed_data_elements $group_pos $loss_weight $in_cnn_out_kp $ShuffleFlag $aug $start_gi
#run_train 4Vm $bs $num_gpus $feed_data_elements $group_pos $loss_weight $in_cnn_out_kp $ShuffleFlag $aug $start_gi

