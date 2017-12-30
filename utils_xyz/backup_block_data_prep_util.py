
class MAIN_DATA_PREP():

    def __init__(self):
        print('Init Class MAIN_DATA_PREP')

    def Do_merge_blocks(self,file_list,stride=[4,4,4],step=[8,8,8]):
        #file_list = glob.glob( os.path.join(GLOBAL_PARA.ETH_A_step_0d5_stride_0d5,   '*_step_0d5_stride_0d5.h5') )
        #file_list = glob.glob( os.path.join(GLOBAL_PARA.ETH_A_stride_1_step_1,   '*_4096.h5') )
        #file_list = glob.glob( os.path.join(GLOBAL_PARA.ETH_A_step_10_stride_10,   '*_blocked.h5_sorted_step_10_stride_10.hdf5') )
        block_step = (np.array(step)).astype(np.int)
        block_stride = (np.array(stride)).astype(np.int)
        #block_stride = (block_step*0.5).astype(np.int)
        print('step = ',block_step)
        print('stride = ',block_stride)

        IsMulti_merge = True
        if not IsMulti_merge:
            for file_name in file_list:
                merged_name = self.merge_blocks_to_new_step(file_name,block_step,block_stride)
                merged_names.append(merged_name)
        else:
            pool = []
            for file_name in file_list:
                p = mp.Process( target=self.merge_blocks_to_new_step, args=(file_name,block_step,block_stride,) )
                p.start()
                pool.append(p)
            for p in pool:
                p.join()

    def merge_blocks_to_new_step(self,base_file_name,larger_step,larger_stride):
        '''merge blocks of sorted raw h5f to get new larger step
        '''
        #new_name = base_file_name.split('_xyz_intensity_rgb')[0] + '_step_' + str(larger_step[0]) + '_stride_' + str(larger_stride[0]) + '.hdf5'
        tmp = rm_file_name_midpart(base_file_name,'_stride_1_step_1')
        new_part = '_stride_' + str(larger_stride[0])+ '_step_' + str(larger_step[0])
        if larger_step[2] != larger_step[0]:
            if larger_step[2]>0:
                new_part += '_z' + str(larger_step[2])
            else:
                new_part += '_zall'

        new_name = os.path.splitext(tmp)[0]  + new_part + '.h5'
        print('new file: ',new_name)
        print('id = ',os.getpid())
        with h5py.File(new_name,'w') as new_h5f:
                base_sh5f = Sorted_H5f(base_h5f,base_file_name)
                new_sh5f = Sorted_H5f(new_h5f,new_name)
                new_sh5f.copy_root_summaryinfo_from_another(self.h5f,'new_stride')
                new_sh5f.set_step_stride(larger_step,larger_stride)

                read_row_N = 0
                rate_last = -10
                print('%d rows and %d blocks to merge'%(base_sh5f.total_row_N,base_sh5f.total_block_N))
                for dset_name in  self.h5f:
                    block_i_base = int(dset_name)
                    base_dset_i = self.h5f[dset_name]
                    block_k_new_ls,i_xyz_new_ls = base_sh5f.get_sub_block_ks(block_i_base,new_sh5f)

                    read_row_N += base_dset_i.shape[0]
                    rate = 100.0 * read_row_N / base_sh5f.total_row_N
                    if int(rate)%10 < 1 and rate-rate_last>5:
                        rate_last = rate
                        print(str(rate),'%   ','  dset_name = ',dset_name, '  new_k= ',block_k_new_ls,'   id= ',os.getpid())
                        new_sh5f.h5f.flush()

                    for block_k_new in block_k_new_ls:
                        new_sh5f.append_to_dset(block_k_new,base_dset_i)
                    #if rate > 5:
                        #break
                if read_row_N != base_sh5f.total_row_N:
                    print('ERROR!!!  total_row_N = %d, but only read %d'%(base_sh5f.total_row_N,read_row_N))

                total_block_N = 0
                total_row_N = 0
                for total_block_N,dn in enumerate(new_sh5f.h5f):
                    total_row_N += new_sh5f.h5f[dn].shape[0]
                total_block_N += 1
                new_sh5f.h5f.attrs['total_row_N']=total_row_N
                new_sh5f.h5f.attrs['total_block_N']=total_block_N
                print('total_row_N = ',total_row_N)
                print('total_block_N = ',total_block_N)
                new_sh5f.h5f.flush()

                #new_sh5f.check_xyz_scope()

                if 'sample_merged' in self.actions:
                    Is_gen_obj = 'obj_sampled_merged' in self.actions
                    Is_gen_norm = 'norm_sampled_merged' in self.actions
                    new_sh5f.file_random_sampling(self.sample_num,self.sample_method,\
                                         gen_norm=Is_gen_norm,gen_obj = Is_gen_obj)


    def gen_rawETH_to_h5(self,label_files_glob,line_num_limit=None):
        '''
        transform the data and label to h5 format
        put every dim to a single dataset
            to speed up search and compare of a single dim
        data is large, chunk to speed up slice
        '''

        label_files_list = glob.glob(label_files_glob)
        data_files_list, h5_files_list = self.clean_label_files_list(label_files_list)
        print('%d data-label files detected'%(len(label_files_list)))
        for lf in label_files_list:
            print('\t%s'%(lf))

        for i,label_fn in enumerate(label_files_list):
            data_fn = data_files_list[i]
            h5_fn = h5_files_list[i]
            with open(data_fn,'r') as data_f:
                with open(label_fn,'r') as label_f:
                    with h5py.File(h5_fn,'w') as h5_f:
                        raw_h5f = Raw_H5f(h5_f,h5_fn)
                        raw_h5f.set_num_default_row(GLOBAL_PARA.h5_num_row_1G)
                        data_label_fs = itertools.izip(data_f,label_f)
                        buf_rows = GLOBAL_PARA.h5_num_row_10M*5
                        data_buf = np.zeros((buf_rows,7),np.float32)
                        label_buf = np.zeros((buf_rows,1),np.int8)
                        for k,data_label_line in enumerate(data_label_fs):
                            k_buf = k%buf_rows
                            data_buf[k_buf,:] =np.fromstring( data_label_line[0].strip(),dtype=np.float32,sep=' ' )
                            label_buf[k_buf,:] = np.fromstring( data_label_line[1].strip(),dtype=np.float32,sep=' ' )
                            if k_buf == buf_rows-1:
                                start = int(k/buf_rows)*buf_rows
                                end = k+1
                                print('start = %d, end = %d in file: %s'%(start,end,data_fn))
                                raw_h5f.add_to_dset('xyz',data_buf[:,0:3],start,end)
                                raw_h5f.add_to_dset('intensity',data_buf[:,3:4],start,end)
                                raw_h5f.add_to_dset('color',data_buf[:,4:7],start,end)
                                raw_h5f.add_to_dset('label',label_buf[:,0:1],start,end)
                                h5_f.flush()

                            if line_num_limit != None and k+1 >= line_num_limit:
                                print('break at k= ',k)
                                break

                        self.add_to_dset_all(raw_h5f,data_buf,label_buf,k,buf_rows)
                        raw_h5f.create_done()

                        print('having read %d lines from %s \n'%(k+1,data_fn))
                        #print('h5 file line num = %d'%(xyz_dset.shape[0]))

    def add_to_dset_all(self,raw_h5f,data_buf,label_buf,k,buf_rows):
        k_buf = k%buf_rows
        start = int(k/buf_rows)*buf_rows
        end = k+1
        #print( 'start = %d, end = %d'%(start,end))
        raw_h5f.add_to_dset('xyz',data_buf[0:k_buf+1,0:3],start,end)
        raw_h5f.add_to_dset('intensity',data_buf[0:k_buf+1,3:4],start,end)
        raw_h5f.add_to_dset('color',data_buf[0:k_buf+1,4:7],start,end)
        raw_h5f.add_to_dset('label',label_buf[0:k_buf+1,0:1],start,end)
        raw_h5f.raw_h5f.flush()
        #print('flushing k = ',k)

    def clean_label_files_list(self,label_files_list):
        data_files_list = []
        h5_files_list = []
        for i,label_file_name in enumerate(label_files_list):
            no_format_name = os.path.splitext(label_file_name)[0]
            data_file_name = no_format_name + '.txt'
            h5_file_name = no_format_name + '.hdf5'
            if not os.path.exists(data_file_name):
                label_files_list.pop(i)
                print('del label_files_list[%d]:%s'%(i,label_file_name))
            else:
                data_files_list.append(data_file_name)
                h5_files_list.append(h5_file_name)
        return data_files_list, h5_files_list


    def DO_add_geometric_scope_file(self):
        files_glob = os.path.join(self.ETH_training_partBh5_folder,'*.hdf5')
        #files_glob = os.path.join(self.ETH_training_partAh5_folder,'*.hdf5')
        #files_glob = os.path.join(self.Local_training_partAh5_folder,'*.hdf5')
        files_list = glob.glob(files_glob)
        print('%d files detected'%(len(files_list)))

        IsMultiProcess = False
        line_num_limit = 1000*100
        line_num_limit = None

        if not IsMultiProcess:
            for file_name in files_list:
                with h5py.File(file_name,'a') as h5f:
                    raw_h5f = Raw_H5f(h5f,file_name)
                    raw_h5f.add_geometric_scope(line_num_limit)
                #self.add_geometric_scope_file(file_name,line_num_limit)
        else:
            mp_n = min(len(files_list),mp.cpu_count())
            pool = mp.Pool(mp_n)
            pool.imap_unordered(self.add_geometric_scope_file,files_list)


    def DO_gen_rawETH_to_h5(self,ETH_raw_labels_glob=None):
        if ETH_raw_labels_glob == None:
            labels_folder = '/home/x/Research/Dataset/ETH_Semantic3D_Dataset/training/part_A'
            labels_folder = '/short/dh01/yx2146/Dataset/ETH_Semantic3D_Dataset/training/part_B'
            #labels_folder = '/other/ETH_Semantic3D_Dataset/training/part_A
            ETH_raw_labels_glob = os.path.join(labels_folder,'*.labels')
        line_num_limit = None
        self.gen_rawETH_to_h5(ETH_raw_labels_glob)


    def main(self,file_list,actions,sample_num=4096,sample_method='random',stride=[4,4,100],step=[8,8,100]):
        # self.actions: [
        # 'merge','sample_merged','obj_sampled_merged','norm_sampled_merged' ]
        self.actions = actions
        self.sample_num = sample_num
        self.sample_method = sample_method
        self.stride = stride
        self.step = step
        if 'merge' in self.actions:
            self.Do_merge_blocks(file_list,self.stride,self.step)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


