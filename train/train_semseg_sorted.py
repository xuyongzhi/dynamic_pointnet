import pdb, traceback
import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import socket
import time
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,'utils'))
sys.path.append(os.path.join(ROOT_DIR,'utils_xyz'))
sys.path.append(os.path.join(ROOT_DIR,'models'))
sys.path.append(os.path.join(ROOT_DIR,'scannet'))
import provider
import get_dataset
from evaluation import EvaluationMetrics
from block_data_net_provider import Normed_H5f,Net_Provider
import multiprocessing as mp
from ply_util import create_ply_matterport, test_box

ISSUMMARY = True
DEBUG_MULTIFEED=False
DEBUG_SMALLDATA=False
IS_GEN_PLY = True
Is_REPORT_PRED = False
ISNoEval = True
LOG_TYPE = 'simple'

parser = argparse.ArgumentParser()
parser.add_argument('--model_flag', default='3A', help='model flag')
parser.add_argument('--model_type', default='presg', help='fds or presg')
parser.add_argument('--dataset_name', default='matterport3d', help='dataset_name: scannet, stanford_indoor,matterport3d')
parser.add_argument('--all_fn_globs', type=str,default='v1/small_test/stride_0d1_step_0d1_pl_nh5_1d6_2/',\
                    help='The file name glob for both training and evaluation')
#parser.add_argument('--all_fn_globs', type=str,default='v1/each_hosue/stride_0d1_step_0d1_pl_nh5_1d6_2/1',\
#                    help='The file name glob for both training and evaluation')
parser.add_argument('--eval_fnglob_or_rate',  default=0.5, help='file name str glob or file number rate: scan1*.nh5 0.2')
parser.add_argument('--bxmh5_folder_name', default='stride_0d1_step_0d1_bmap_nh5_25600_1d6_2_fmn3-512_256_64-128_12_6-0d2_0d6_1d2-0d2_0d6_1d2', help='')
parser.add_argument('--feed_data_elements', default='xyz-color_1norm', help='xyz_1norm_file-xyz_midnorm_block-color_1norm')
#parser.add_argument('--feed_data_elements', default='xyz_1norm_block-color_1norm', help='xyz_1norm_file-xyz_midnorm_block-color_1norm')
#parser.add_argument('--feed_data_elements', default='xyz_midnorm_block-color_1norm', help='xyz_1norm_file-xyz_midnorm_block-color_1norm')
#parser.add_argument('--feed_data_elements', default='xyz-color_1norm', help='xyz_1norm_file-xyz_midnorm_block-color_1norm')
parser.add_argument('--feed_label_elements', default='label_category', help='label_category-label_instance')
parser.add_argument('--batch_size', type=int, default=9, help='Batch Size during training [default: 24]')
parser.add_argument('--num_point', type=int, default=-1, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epoch to run [default: 50]')

parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--learning_rate', type=float, default=0.005, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--max_test_file_num', type=int, default=None, help='Which area to use for test, option: 1-6 [default: 6]')

parser.add_argument('--only_evaluate',action='store_true',help='do not train')
parser.add_argument('--finetune',action='store_true',help='do not train')
parser.add_argument('--model_epoch', type=int, default=10, help='the epoch of model to be restored')

parser.add_argument('--auto_break',action='store_true',help='If true, auto break when error occurs')
parser.add_argument('--debug',action='store_true',help='tf debug')
parser.add_argument('--multip_feed',action='store_true',help='IsFeedData_MultiProcessing = True')

FLAGS = parser.parse_args()

#-------------------------------------------------------------------------------
ISDEBUG = FLAGS.debug

if IS_GEN_PLY:
    FLAGS.feed_data_elements = 'xyz-color_1norm'
    #FLAGS.feed_data_elements = 'xyz_1norm_block-color_1norm'
    FLAGS.max_epoch = 1
    FLAGS.finetune = True
    FLAGS.model_epoch = 199
    #FLAGS.batch_size = 1

#FLAGS.finetune = True
#FLAGS.model_epoch = 99
#-------------------------------------------------------------------------------
feed_data_elements = FLAGS.feed_data_elements.split('-')
feed_label_elements = FLAGS.feed_label_elements.split('-')
if FLAGS.model_type == 'fds':
    from pointnet2_sem_seg import  get_model,get_loss
    import pdb; pdb.set_trace()
    from pointnet2_sem_seg import placeholder_inputs
elif FLAGS.model_type == 'presg':
    from pointnet2_sem_seg_presg import  placeholder_inputs,get_model,get_loss

BATCH_SIZE = FLAGS.batch_size
#FLAGS.num_point = Net_Provider.global_num_point
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate


# ------------------------------------------------------------------------------
# Load Data
FLAGS.all_fn_globs = FLAGS.all_fn_globs.split(',')
net_provider = Net_Provider(
                            dataset_name=FLAGS.dataset_name,
                            all_filename_glob=FLAGS.all_fn_globs,
                            eval_fnglob_or_rate=FLAGS.eval_fnglob_or_rate,
                            bxmh5_folder_name = FLAGS.bxmh5_folder_name,
                            only_evaluate = FLAGS.only_evaluate,
                            feed_data_elements=feed_data_elements,
                            feed_label_elements=feed_label_elements)

NUM_POINT = net_provider.global_num_point
NUM_DATA_ELES = net_provider.data_num_eles
NUM_CLASSES = net_provider.num_classes
NUM_LABEL_ELES = net_provider.label_num_eles
LABEL_ELE_IDXS = net_provider.feed_label_ele_idxs
DATA_ELE_IDXS = net_provider.feed_data_ele_idxs
CATEGORY_LABEL_IDX = LABEL_ELE_IDXS['label_category'][0]
TRAIN_FILE_N = net_provider.train_file_N
EVAL_FILE_N = net_provider.eval_file_N
MAX_MULTIFEED_NUM = 5

# ------------------------------------------------------------------------------
try:
    FLAGS.eval_fnglob_or_rate = float(FLAGS.eval_fnglob_or_rate)
    log_eval_fn_glob = ''
    print('FLAGS.eval_fnglob_or_rate is eval file number rate')
except:
    log_eval_fn_glob = FLAGS.eval_fnglob_or_rate.split('*')[0]
    print('FLAGS.eval_fnglob_or_rate is eval name glob. log_eval_fn_glob:%s'%(log_eval_fn_glob))

if FLAGS.only_evaluate:
    MAX_EPOCH = 1
    log_name = 'log_test.txt'
else:
    MAX_EPOCH = FLAGS.max_epoch
    log_name = 'log_train.txt'
    gsbb_config = net_provider.gsbb_config
    FLAGS.log_dir = FLAGS.log_dir+'-model_'+FLAGS.model_flag+'-gsbb_'+gsbb_config+'-bs'+str(BATCH_SIZE)+'-'+\
                    FLAGS.feed_data_elements+'-'+str(NUM_POINT)+'-'+FLAGS.dataset_name[0:3]

LOG_DIR = os.path.join(ROOT_DIR,'train_res/semseg_result/'+FLAGS.log_dir)
MODEL_PATH = os.path.join(LOG_DIR,'model.ckpt-'+str(FLAGS.model_epoch))
LOG_DIR_FUSION = os.path.join(ROOT_DIR,'train_res/semseg_result/fusion_log.txt')
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
os.system('cp %s/models/pointnet2_sem_seg.py %s' % (ROOT_DIR,LOG_DIR)) # bkp of model def
os.system('cp %s/train_semseg_sorted.py %s' % (BASE_DIR,LOG_DIR)) # bkp of train procedure
if FLAGS.finetune:
    LOG_FOUT = open(os.path.join(LOG_DIR, log_name), 'a')
else:
    LOG_FOUT = open(os.path.join(LOG_DIR, log_name), 'w')
LOG_FOUT_FUSION = open(LOG_DIR_FUSION, 'a')
LOG_FOUT.write(str(FLAGS)+'\n\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.6
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

print('BATCH_SIZE = %d'%(BATCH_SIZE))
# ------------------------------------------------------------------------------
BLOCK_SAMPLE = net_provider.block_sample
if  DEBUG_SMALLDATA:
    LIMIT_MAX_NUM_BATCHES = {}
    LIMIT_MAX_NUM_BATCHES['train'] = 4
    LIMIT_MAX_NUM_BATCHES['test'] = 2

START_TIME = time.time()

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(global_step):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        global_step * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate

def get_bn_decay(global_step):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      global_step * BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train_eval(train_feed_buf_q, train_multi_feed_flags, eval_feed_buf_q, eval_multi_feed_flags, lock):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            #pointclouds_pl, labels_pl,smpws_pl = placeholder_inputs(BATCH_SIZE,BLOCK_SAMPLE,NUM_DATA_ELES,NUM_LABEL_ELES)
            flatten_bm_extract_idx = net_provider.flatten_bidxmaps_extract_idx
            pointclouds_pl, labels_pl, smpws_pl,  sg_bidxmaps_pl, flatten_bidxmaps_pl = placeholder_inputs(BATCH_SIZE,BLOCK_SAMPLE,
                                        NUM_DATA_ELES,NUM_LABEL_ELES,net_provider.sg_bidxmaps_shape,net_provider.flatten_bidxmaps_shape, flatten_bm_extract_idx )
            category_labels_pl = labels_pl[...,CATEGORY_LABEL_IDX]
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=global_step parameter to minimize.
            # That tells the optimizer to helpfully increment the 'global_step' parameter for you every time it trains.
            global_step = tf.Variable(0)
            bn_decay = get_bn_decay(global_step)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            if FLAGS.model_type == 'fds':
                pred,end_points = get_model( pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
                loss = get_loss(pred, labels_pl,smpws_pl)
            elif FLAGS.model_type == 'presg':
                sg_bm_extract_idx = net_provider.sg_bidxmaps_extract_idx
                pred, end_points, debug = get_model( FLAGS.model_flag, pointclouds_pl, is_training_pl, NUM_CLASSES, sg_bidxmaps_pl, sg_bm_extract_idx, flatten_bidxmaps_pl, flatten_bm_extract_idx, bn_decay=bn_decay)
                loss = get_loss(pred, labels_pl, smpws_pl, LABEL_ELE_IDXS)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(category_labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(global_step)
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('global_step', global_step)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=global_step)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver(max_to_keep=50)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.InteractiveSession(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        if not FLAGS.only_evaluate:
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                    sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))
        else:
            test_writer = None

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})
        if ISDEBUG:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan",tf_debug.has_inf_or_nan)

        ops = {'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': global_step,
               'accuracy':accuracy}
        ops['pointclouds_pl'] = pointclouds_pl
        ops['labels_pl'] = labels_pl
        ops['smpws_pl'] = smpws_pl
        ops['sg_bidxmaps_pl'] = sg_bidxmaps_pl
        ops['flatten_bidxmaps_pl'] = flatten_bidxmaps_pl
        if 'l_xyz' in debug:
            ops['l_xyz'] = debug['l_xyz']
        if 'grouped_xyz' in debug:
            ops['grouped_xyz'] = debug['grouped_xyz']
            ops['flat_xyz'] = debug['flat_xyz']
            ops['flatten_bidxmap'] = debug['flatten_bidxmap']

        if FLAGS.finetune:
            saver.restore(sess,MODEL_PATH)
            log_string('finetune, restored model from: \n\t%s'%MODEL_PATH)

        log_string(net_provider.data_summary_str)


        epoch_start = 0
        if FLAGS.finetune:
            epoch_start+=(FLAGS.model_epoch+1)
        for epoch in range(epoch_start,epoch_start+MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            if train_feed_buf_q == None:
                net_provider.update_train_eval_shuffled_idx()
            if not FLAGS.only_evaluate:
                train_log_str = train_one_epoch(sess, ops, train_writer,epoch,train_feed_buf_q, train_multi_feed_flags, lock)
            else:
                train_log_str = ''
                saver.restore(sess,MODEL_PATH)
                log_string('only evaluate, restored model from: \n\t%s'%MODEL_PATH)
            if ISNoEval: eval_log_str = ''
            else:
                eval_log_str = eval_one_epoch(sess, ops, test_writer,epoch, eval_feed_buf_q, eval_multi_feed_flags, lock )

            # Save the variables to disk.
            if not FLAGS.only_evaluate:
                if (epoch > 0 and epoch % 1 == 0) or epoch == MAX_EPOCH-1:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"),global_step=epoch)
                    log_string("Model saved in file: %s" % os.path.basename(save_path))

            if epoch == MAX_EPOCH -1:
                LOG_FOUT_FUSION.write( str(FLAGS)+'\n\n'+train_log_str+'\n'+eval_log_str+'\n\n' )

            #print('train eval finish epoch %d / %d'%(epoch,epoch_start+MAX_EPOCH-1))

def add_log(tot,epoch,batch_idx,loss_batch,t_batch_ls,SimpleFlag = 0,c_TP_FN_FP = None,total_seen=None,accuracy=None):

    if accuracy != None:
        ave_whole_acc = accuracy
    else:
        ave_whole_acc,class_acc_str,ave_acc_str = EvaluationMetrics.get_class_accuracy(
                                    c_TP_FN_FP,total_seen)
    log_str = ''
    if len(t_batch_ls)>0:
        t_per_batch = np.mean(np.concatenate(t_batch_ls,axis=1),axis=1)
        t_per_block = t_per_batch / BATCH_SIZE
        t_per_block_str = np.array2string(t_per_block*1000,formatter={'float_kind':lambda x: "%0.1f"%x})
    else:
        t_per_block_str = "no-t"
    log_str += '%s [%d - %d] \t t_block(d,c):%s\tloss: %0.3f \tacc: %0.3f' % \
            ( tot,epoch,batch_idx,t_per_block_str,loss_batch,ave_whole_acc )
    if accuracy == None:
        if SimpleFlag >0:
            log_str += ave_acc_str
        if  SimpleFlag >1:
            log_str += class_acc_str
    log_string(log_str)
    return log_str

def train_one_epoch(sess, ops, train_writer,epoch,train_feed_buf_q, train_multi_feed_flags, lock):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    #log_string('----')
    num_blocks = net_provider.train_num_blocks
    if num_blocks!=None:
        num_batches = num_blocks // BATCH_SIZE
        if DEBUG_SMALLDATA: num_batches = min(num_batches,LIMIT_MAX_NUM_BATCHES['train'])
        assert num_batches >0, "num_batches = 0, num_blocks = %d, BATCH_SIZE = %d"%(num_blocks,BATCH_SIZE)
    else:
        num_batches = None

    total_seen = 0.0001
    loss_sum = 0.0
    accuracy_sum = 0.0
    c_TP_FN_FP = np.zeros(shape=(3,NUM_CLASSES))

    print('total batch num = ',num_batches)
    batch_idx = -1

    t_batch_ls=[]
    train_logstr = ''
    while ( train_feed_buf_q!=None ) or  ( batch_idx < num_batches-1) or (num_batches==None):
        # When use multi feed, stop with train_feed_thread_finish_num.value
        # When use normal feed, stop with batch_idx
        t0 = time.time()
        batch_idx += 1
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        if train_feed_buf_q == None:
            IsShuffleIdx = epoch%2 != 0
            cur_data,cur_label,cur_smp_weights,cur_sg_bidxmaps,cur_flatten_bidxmaps = net_provider.get_train_batch(start_idx,end_idx,IsShuffleIdx)
        else:
            if train_feed_buf_q.qsize() == 0:
                if train_multi_feed_flags['feed_finish_epoch'].value == epoch:
                    with lock:
                        train_multi_feed_flags['read_OK_epoch'].value = epoch
                    if DEBUG_MULTIFEED: print('train read OK, epoch=%d  batch_idx=%d'%(epoch,batch_idx))
                    break

                bufread_t0 = time.time()
                while train_feed_buf_q.qsize() == 0:
                    #print('no data in train_feed_buf_q')
                    if time.time() - bufread_t0 > 5:
                        print('\nWARING!!! no data in train_feed_buf_q for long time, epoch=%d   batch_idx=%d\n'%(epoch,batch_idx))
                        bufread_t0 = time.time()
                    time.sleep(0.1)
            #if DEBUG_MULTIFEED: print('get train_feed_buf_q size= %d,  batch_idx=%d'%(train_feed_buf_q.qsize(),batch_idx))
            cur_data,cur_label,cur_smp_weights, cur_sg_bidxmaps, cur_flatten_bidxmaps,  batch_idx_buf,epoch_buf = train_feed_buf_q.get()
            #assert batch_idx == batch_idx_buf and epoch== epoch_buf
        #if DEBUG_MULTIFEED: continue

        t1 = time.time()
        if type(cur_data) == type(None):
            break # all data reading finished
        label_category = cur_label[:,:,CATEGORY_LABEL_IDX]
        feed_dict = { ops['is_training_pl']: is_training }
        feed_dict[ops['pointclouds_pl']] = cur_data
        feed_dict[ops['labels_pl']] = cur_label
        feed_dict[ops['smpws_pl']] = cur_smp_weights
        feed_dict[ops['sg_bidxmaps_pl']] = cur_sg_bidxmaps
        feed_dict[ops['flatten_bidxmaps_pl']] = cur_flatten_bidxmaps

        summary, step, _, loss_val, pred_val, accuracy_batch = sess.run( [ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred'], ops['accuracy']],
                                    feed_dict=feed_dict )

        cur_label, = sess.run( [ops['labels_pl']], feed_dict=feed_dict )
        if IS_GEN_PLY and batch_idx<3:
            color_flags = ['gt_color']
            gen_ply( batch_idx, cur_data[:,:,0:3], color_flags,  cur_label, np.argmax(pred_val,2), cur_data,name_meta = '_rawdata')
            #pl_display, = sess.run( [ops['pointclouds_pl']], feed_dict=feed_dict )
            for lk in range( len(ops['l_xyz']) ):
                pl_display, = sess.run( [ops['l_xyz'][lk]], feed_dict=feed_dict )
                if lk == 0:
                    color_flags = ['gt_color']
                    gen_ply( batch_idx,pl_display, color_flags,  cur_label, np.argmax(pred_val,2), cur_data,name_meta = '_lxyz0')
                else:
                    color_flags = ['no_color']
                    gen_ply( batch_idx, pl_display, color_flags,  name_meta = '_lxyz'+str(lk))
            for lk in range( len(ops['grouped_xyz']) ):
                grouped_xyz, flat_xyz, flatten_bidxmap = sess.run( [ops['grouped_xyz'][lk], ops['flat_xyz'][lk], ops['flatten_bidxmap'][lk] ], feed_dict=feed_dict )
                color_flags = ['no_color']
                gen_ply( batch_idx, grouped_xyz, color_flags, name_meta = '_gpxyz'+str(lk))
                gen_ply( batch_idx, flat_xyz, color_flags, name_meta = '_flatxyz'+str(lk))
                missed_b_num = np.sum( flatten_bidxmap[...,0,1] < 0 )
                print('missed_b_num:', missed_b_num)
                import pdb; pdb.set_trace()  # XXX BREAKPOINT

        if Is_REPORT_PRED:
            pred_fn = LOG_DIR+'/train_pred_log.txt'
            EvaluationMetrics.report_pred( pred_fn, pred_val, cur_label[...,CATEGORY_LABEL_IDX], 'matterport3d' )

        loss_sum += loss_val
        accuracy_sum += accuracy_batch
        t_batch_ls.append( np.reshape(np.array([t1-t0,time.time() - t1]),(2,1)) )
        if ISSUMMARY: train_writer.add_summary(summary, step)
        if batch_idx == num_batches-1 or  (epoch == 0 and batch_idx % 20 ==0) or (batch_idx%50==0):
            if LOG_TYPE == 'complex':
                pred_val = np.argmax(pred_val, 2)
                total_seen += (BATCH_SIZE*NUM_POINT)
                c_TP_FN_FP += EvaluationMetrics.get_TP_FN_FP(NUM_CLASSES,pred_val,cur_label[...,CATEGORY_LABEL_IDX])
                train_logstr = add_log('train',epoch,batch_idx,loss_sum/(batch_idx+1),t_batch_ls,c_TP_FN_FP = c_TP_FN_FP,total_seen = total_seen)
            else:
                train_logstr = add_log('train',epoch,batch_idx,loss_sum/(batch_idx+1),t_batch_ls,accuracy = accuracy_sum/(batch_idx+1))
        if batch_idx == 200:
            os.system('nvidia-smi')
    print('train epoch %d finished, batch_idx=%d'%(epoch,batch_idx))
    return train_logstr

def gen_ply(batch_idx, pl_display, color_flags = ['gt_color'], cur_label=None, pred_val=None, raw_data=None, name_meta = ''):
    #color_flags = ['raw_color']
    #color_flags = ['gt_color']
    position = 'xyz_midnorm_block'
    position = 'xyz_1norm_block'
    position = 'xyz'
    if position!='xyz':
        assert BATCH_SIZE == 1
    if 'no_color' in color_flags:
        cur_xyz = pl_display[...,DATA_ELE_IDXS[position]]
        create_ply_matterport( cur_xyz, LOG_DIR+'/train_%d_nocolor'%(batch_idx)+name_meta+'.ply' )

    if 'gt_color' in color_flags:
        cur_xyz = pl_display[...,DATA_ELE_IDXS[position]]
        cur_label_category = cur_label[...,CATEGORY_LABEL_IDX]
        create_ply_matterport( cur_xyz, LOG_DIR+'/train_%d_gtcolor'%(batch_idx)+name_meta+'.ply', cur_label_category  )
        create_ply_matterport( cur_xyz, LOG_DIR+'/train_%d_predcolor'%(batch_idx)+name_meta+'.ply', pred_val )
        err_idxs = cur_label_category != pred_val
        create_ply_matterport( cur_xyz[err_idxs], LOG_DIR+'/train_%d_err_predcolor'%(batch_idx)+name_meta+'.ply', pred_val[err_idxs] )
        create_ply_matterport( cur_xyz[err_idxs], LOG_DIR+'/train_%d_err_gtcolor'%(batch_idx)+name_meta+'.ply', cur_label_category[err_idxs] )

    if 'raw_color' in color_flags:
        cur_xyz_color = pl_display[...,DATA_ELE_IDXS[position]+DATA_ELE_IDXS['color_1norm']]
        cur_xyz_color[...,[3,4,5]] *= 255
        create_ply_matterport( cur_xyz_color, LOG_DIR+'/train_%d_rawcolor'%(batch_idx)+'.ply' )
        cur_xyz_color = raw_data[...,DATA_ELE_IDXS[position]+DATA_ELE_IDXS['color_1norm']]
        cur_xyz_color[...,[3,4,5]] *= 255
        create_ply_matterport( cur_xyz_color, LOG_DIR+'/train_grouped_%d_rawcolor'%(batch_idx)+'.ply' )

def limit_eval_num_batches(epoch,num_batches):
    if epoch%5 != 0:
        num_batches = min(num_batches,31)
    return num_batches

def eval_one_epoch(sess, ops, test_writer, epoch, eval_feed_buf_q, eval_multi_feed_flags, lock):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_seen = 0.00001
    loss_sum = 0.0
    accuracy_sum = 0.0
    c_TP_FN_FP = np.zeros(shape=(3,NUM_CLASSES))

    log_string('----')

    num_blocks = net_provider.eval_num_blocks
    if num_blocks != None:
        num_batches = num_blocks // BATCH_SIZE
        #num_batches = limit_eval_num_batches(epoch,num_batches)
        if DEBUG_SMALLDATA: num_batches = min(num_batches,LIMIT_MAX_NUM_BATCHES['test'])
        if num_batches == 0:
            print('\ntest num_blocks=%d  BATCH_SIZE=%d  num_batches=%d'%(num_blocks,BATCH_SIZE,num_batches))
            return ''
    else:
        num_batches = None

    eval_logstr = ''
    t_batch_ls = []
    batch_idx = -1

    while (eval_feed_buf_q!=None) or (batch_idx < num_batches-1) or (num_batches==None):
        t0 = time.time()
        batch_idx += 1
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        if eval_feed_buf_q == None:
            cur_data,cur_label,cur_smp_weights,cur_sg_bidxmaps,cur_flatten_bidxmaps  = net_provider.get_eval_batch(start_idx,end_idx,False)
        else:
            if eval_feed_buf_q.qsize() == 0:
                if eval_multi_feed_flags['feed_finish_epoch'].value == epoch:
                    with lock:
                        eval_multi_feed_flags['read_OK_epoch'].value = epoch
                    #if DEBUG_MULTIFEED: print('eval read OK, epoch=%d  batch_idx=%d'%(epoch,batch_idx))
                    break

            bufread_t0 = time.time()
            while eval_feed_buf_q.qsize() == 0:
                if time.time() - bufread_t0 > 2:
                    print('\nWARING!!! no data in eval_feed_buf_q for long time\n')
                    bufread_t0 = time.time()
                time.sleep(0.2)
            #print('eval_feed_buf_q size= %d'%(eval_feed_buf_q.qsize()))

            cur_data,cur_label,cur_smp_weights, cur_sg_bidxmaps, cur_flatten_bidxmaps, batch_idx_buf,epoch_buf  = eval_feed_buf_q.get()
            #assert batch_idx == batch_idx_buf and epoch== epoch_buf

        #if DEBUG_MULTIFEED: continue

        t1 = time.time()
        if type(cur_data) == type(None):
            print('batch_idx:%d, get None, reading finished'%(batch_idx))
            break # all data reading finished

        feed_dict = { ops['is_training_pl']: is_training }
        feed_dict[ops['pointclouds_pl']] = cur_data
        feed_dict[ops['labels_pl']] = cur_label
        feed_dict[ops['smpws_pl']] = cur_smp_weights
        feed_dict[ops['sg_bidxmaps_pl']] = cur_sg_bidxmaps
        feed_dict[ops['flatten_bidxmaps_pl']] = cur_flatten_bidxmaps

        if FLAGS.datafeed_type == 'Pr_Normed_H5f':
            cur_label, = sess.run( [ops['labels_pl']], feed_dict=feed_dict )
        summary, step, loss_val, pred_val,accuracy_batch = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred'],ops['accuracy']],
                                      feed_dict=feed_dict)
        if ISSUMMARY and  test_writer != None:
            test_writer.add_summary(summary, step)
        t_batch_ls.append( np.reshape(np.array([t1-t0,time.time() - t1]),(2,1)) )

        accuracy_sum += accuracy_batch
        loss_sum += loss_val
        if batch_idx == num_batches-1 or (FLAGS.only_evaluate and  batch_idx%30==0):
            if LOG_TYPE == 'complex':
                pred_logits = np.argmax(pred_val, 2)
                total_seen += (BATCH_SIZE*NUM_POINT)
                c_TP_FN_FP += EvaluationMetrics.get_TP_FN_FP(NUM_CLASSES,pred_logits,cur_label[...,CATEGORY_LABEL_IDX])
                #net_provider.set_pred_label_batch(pred_val,start_idx,end_idx)
                eval_logstr = add_log('eval',epoch,batch_idx,loss_sum/(batch_idx+1),t_batch_ls,c_TP_FN_FP = c_TP_FN_FP,total_seen = total_seen)
            else:
                eval_logstr = add_log('eval',epoch,batch_idx,loss_sum/(batch_idx+1),t_batch_ls,accuracy = accuracy_sum/(batch_idx+1))

    #if FLAGS.only_evaluate:
    #    obj_dump_dir = os.path.join(FLAGS.log_dir,'obj_dump')
    #    net_provider.gen_gt_pred_objs(FLAGS.visu,obj_dump_dir)
    #    net_provider.write_file_accuracies(FLAGS.log_dir)
    #    print('\nobj out path:'+obj_dump_dir)
    print('eval epoch %d finished'%(epoch))
    return eval_logstr


def add_feed_buf(train_or_test,feed_buf_q, cpu_id, file_id_start, file_id_end, multi_feed_flags, lock, limit_max_train_num_batches=None):
    with tf.device('/cpu:%d'%(cpu_id)):
        max_buf_size = 10
        block_idx_start = net_provider.g_block_idxs[file_id_start,0]
        block_idx_end = net_provider.g_block_idxs[file_id_end,1]
        if train_or_test=='test':
            block_idx_start -= net_provider.eval_global_start_idx
            block_idx_end -= net_provider.eval_global_start_idx
        batch_idx_start = int(math.ceil( 1.0 * block_idx_start / BATCH_SIZE ))
        batch_idx_end = block_idx_end // BATCH_SIZE
        num_batches = batch_idx_end - batch_idx_start
        if DEBUG_SMALLDATA and limit_max_train_num_batches!=None: num_batches = min(num_batches,limit_max_train_num_batches)
        #if DEBUG_MULTIFEED: print('%s cpuid=%d  batch_idx: %d - %d'%(train_or_test,cpu_id,batch_idx_start,batch_idx_end))

        epoch_start = 0
        if FLAGS.finetune:
            epoch_start+=(FLAGS.model_epoch+1)
        for epoch in range(epoch_start,epoch_start+MAX_EPOCH):
            while True:
                if multi_feed_flags['read_OK_epoch'].value == epoch-1:
                    break
                if DEBUG_MULTIFEED: print('%s, cpuid=%d, epoch=%d, read_OK_epoch=%d, waiting for computation and reading in other threads finished'%(train_or_test, cpu_id,epoch, multi_feed_flags['read_OK_epoch'].value))
                time.sleep(3)
            IsShuffleIdx = epoch%3 == 0
            #IsShuffleIdx = False
            if cpu_id==0:
                log_string('epoch %d train IsShuffleIdx: %s'%(epoch,IsShuffleIdx))
                if IsShuffleIdx: net_provider.update_train_eval_shuffled_idx()

            batch_idx = -1 + batch_idx_start
            while (batch_idx < num_batches-1 + batch_idx_start) or (num_batches==None):
                if feed_buf_q.qsize() < max_buf_size:
                    batch_idx += 1
                    block_start_idx = batch_idx * BATCH_SIZE
                    block_end_idx = (batch_idx+1) * BATCH_SIZE
                    if train_or_test == 'train':
                        cur_data,cur_label,cur_smp_weights,cur_sg_bidxmaps,cur_flatten_bidxmaps  = net_provider.get_train_batch(block_start_idx,block_end_idx,IsShuffleIdx)
                    elif train_or_test == 'test':
                        cur_data,cur_label,cur_smp_weights,cur_sg_bidxmaps,cur_flatten_bidxmaps  = net_provider.get_eval_batch(block_start_idx,block_end_idx,False)
                    feed_buf_q.put( [cur_data,cur_label,cur_smp_weights, cur_sg_bidxmaps, cur_flatten_bidxmaps, batch_idx,epoch] )
                    if type(cur_data) == type(None):
                        print('add_train_feed_buf: get None data from net_provider, all data put finished. epoch= %d, batch_idx= %d'%(epoch,batch_idx))
                        break # all data reading finished
                    if DEBUG_MULTIFEED: print('put %s feed_buf_q, size=%d, cpu_id=%d, batch_idx=%d'%( train_or_test, feed_buf_q.qsize(),cpu_id,batch_idx))
                else:
                    #if DEBUG_MULTIFEED: print('%s buf full, cpu_id=%d, batch_idx=%d'%(train_or_test, cpu_id, batch_idx))
                    time.sleep(0.2)
            with lock:
                multi_feed_flags['feed_thread_finish_num'].value += 1
                #print('add_feed_buf: %s data reading finished. epoch= %d, batch_idx= %d, num_batches=%d cpu_id=%d, feed_thread_finish_num=%d'%(train_or_test,epoch,batch_idx,num_batches,cpu_id,multi_feed_flags['feed_thread_finish_num'].value))
                if train_or_test == 'train': file_num = TRAIN_FILE_N
                elif train_or_test == 'test': file_num = EVAL_FILE_N
                if multi_feed_flags['feed_thread_finish_num'].value == min(MAX_MULTIFEED_NUM,file_num):
                    multi_feed_flags['feed_thread_finish_num'].value = 0
                    multi_feed_flags['feed_finish_epoch'].value = epoch
                    if DEBUG_MULTIFEED: print('%s feed OK, epoch=%d  batch_idx=%d'%(train_or_test, epoch,batch_idx))

def main():
    IsFeedData_MultiProcessing = FLAGS.multip_feed and (not FLAGS.auto_break)

    if IsFeedData_MultiProcessing:
        feed_buf_qs = {}
        feed_buf_qs['train'] = mp.Queue()
        feed_buf_qs['test'] = mp.Queue()

        processes = {}
        lock = mp.Lock()
        two_multi_feed_flags = {}
        two_multi_feed_flags['train'] = {}
        two_multi_feed_flags['test'] = {}

        two_multi_feed_flags['train']['feed_thread_finish_num'] = mp.Value('i',0)
        two_multi_feed_flags['test']['feed_thread_finish_num'] = mp.Value('i',0)

        if FLAGS.finetune:
            epoch_start = (FLAGS.model_epoch+1)
        else:
            epoch_start = 0
        two_multi_feed_flags['train']['read_OK_epoch'] = mp.Value('i',epoch_start-1)
        two_multi_feed_flags['test']['read_OK_epoch'] = mp.Value('i',epoch_start-1)
        two_multi_feed_flags['train']['feed_finish_epoch'] = mp.Value('i',epoch_start-1)
        two_multi_feed_flags['test']['feed_finish_epoch'] = mp.Value('i',epoch_start-1)

        file_nums = {}
        file_nums['train'] = net_provider.train_file_N
        file_nums['test'] = net_provider.eval_file_N

        for tot in ['train','test']:
            if ISNoEval and tot=='test': continue
            for k in range( min(MAX_MULTIFEED_NUM,file_nums[tot]) ):
                if DEBUG_SMALLDATA: limit_max_train_num_batches = int( max(1, LIMIT_MAX_NUM_BATCHES[tot]/min(file_nums[tot],MAX_MULTIFEED_NUM) ) )
                else: limit_max_train_num_batches = None
                cpu_id = k
                file_id_start = k
                file_id_end = k
                if k == MAX_MULTIFEED_NUM-1:
                    file_id_end = file_nums[tot]-1
                if tot == 'test':
                    cpu_id += file_nums['train']
                    file_id_start += file_nums['train']
                    file_id_end += file_nums['train']
                processes[tot+'_feed_'+str(k)] = mp.Process(target=add_feed_buf,args=(tot, feed_buf_qs[tot], k, file_id_start, file_id_end, two_multi_feed_flags[tot], lock, limit_max_train_num_batches))

        processes[ 'train_eval'] = mp.Process(target=train_eval,args=(feed_buf_qs['train'], two_multi_feed_flags['train'], feed_buf_qs['test'], two_multi_feed_flags['test'], lock))
        for p in processes:
            processes[p].start()
        for p in processes:
            processes[p].join()
    else:
        train_eval(None,None,None,None,None)

if __name__ == "__main__":
    if FLAGS.auto_break:
        try:
            main()
            LOG_FOUT.close()
        except:
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
    else:
        main()
        #train_eval(None,None)
        LOG_FOUT.close()
