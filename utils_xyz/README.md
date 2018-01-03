
- [ ] Add GlobalSubBaseBLOCK configuration check in get_data_larger_block
- [ ] get_blockids_of_dif_stride_step cost too much time
- [ ] Check the correctness of norm in get_block_data_of_new_stride_step_byid
- [ ] Check the performance and time cost fo probability sampling of:
	 sample_choice,_ = get_sample_choice( all_cur_blockids.shape[0],GlobalSubBaseBlock.nsubblock,GlobalSubBaseBlock.nsubblock,random_sampl_pro )
	 in get_data_larger_block
- [ ] Check the performance of randomly sampling of:
	sample_choice,reduced_num = get_sample_choice(datas.shape[0],sample_num) in
	get_block_data_of_new_stride_step_byid
- [ ] Candidate improvement:                                                 
           1) save all_cur_blockids and all_cur_block_size in h5f to save time.
