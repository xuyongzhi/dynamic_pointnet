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
import tf_util
import provider
import get_dataset
from evaluation import EvaluationMetrics
from block_data_net_provider import Normed_H5f,Net_Provider
import multiprocessing as mp
from ply_util import create_ply_matterport, test_box
from time import gmtime, strftime
from configs import NETCONFIG
from pointnet2_sem_seg_presg import  placeholder_inputs,get_model,get_loss

DEBUG_TMP = False
ISSUMMARY = True
DEBUG_MULTIFEED=False
DEBUG_SMALLDATA=False

parser = argparse.ArgumentParser()
parser.add_argument('--modelf_nein', default='4aG_114', help='{model flag}_{neighbor num of cascade 0,0 from 1,and others}')
parser.add_argument('--dataset_name', default='scannet', help='dataset_name: scannet, stanford_indoor,matterport3d')
parser.add_argument('--all_fn_globs', type=str,default='Merged_sph5/90000_gs-4_-6d3/', help='The file name glob for both training and evaluation')
parser.add_argument('--eval_fnglob_or_rate',  default=0, help='file name str glob or file number rate: scan1*.nh5 0.2')
parser.add_argument('--bxmh5_folder_name', default='Merged_bxmh5/90000_gs-4_-6d3_fmn6-6400_2400_320_32-32_16_32_48-0d1_0d3_0d9_2d7-0d1_0d2_0d6_1d8-pd3-4C0', help='')
parser.add_argument('--feed_data_elements', default='xyz_midnorm_block-color_1norm', help='xyz_1norm_file-xyz_midnorm_block-color_1norm')
parser.add_argument('--feed_label_elements', default='label_category', help='label_category-label_instance')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 24]')
parser.add_argument('--num_point', type=int, default=-1, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 50]')
parser.add_argument('--group_pos',default='mean',help='mean or bc(block center)')

parser.add_argument('--num_gpus', type=int, default=2, help='GPU num]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_epoch_step', type=int, default=50, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--max_test_file_num', type=int, default=None, help='Which area to use for test, option: 1-6 [default: 6]')

parser.add_argument('--only_evaluate',type=int,help='do not train')
parser.add_argument('--finetune',type=int,default=0,help='do not train')
parser.add_argument('--model_epoch', type=int, default=10, help='the epoch of model to be restored')

parser.add_argument('--auto_break',action='store_true',help='If true, auto break when error occurs')
parser.add_argument('--debug',action='store_true',help='tf debug')
parser.add_argument('--multip_feed',type=int, default=0,help='IsFeedData_MultiProcessing = True')
parser.add_argument('--ShuffleFlag', default='Y', help='N:no,M:mix,Y:yes')
parser.add_argument('--loss_weight', default='E', help='E: Equal, N:Number, C:Center, CN')
parser.add_argument('--inkp_min', type=float, default=0.3, help='random input drop minimum')
parser.add_argument('--inkp_max', type=float, default=1.0, help='random input drop maxmum')

FLAGS = parser.parse_args()
FLAGS.finetune = bool(FLAGS.finetune)
FLAGS.multip_feed = bool(FLAGS.multip_feed)
FLAGS.only_evaluate = bool(FLAGS.only_evaluate)
IS_GEN_PLY = True and FLAGS.only_evaluate
Is_REPORT_PRED = IS_GEN_PLY
assert FLAGS.ShuffleFlag=='N' or FLAGS.ShuffleFlag=='Y' or FLAGS.ShuffleFlag=='M'
#-------------------------------------------------------------------------------
ISDEBUG = FLAGS.debug
Feed_Data_Elements = FLAGS.feed_data_elements.split('-')
Feed_Label_Elements = FLAGS.feed_label_elements.split('-')
try:
    FLAGS.eval_fnglob_or_rate=float(FLAGS.eval_fnglob_or_rate)
except:
    pass
ISNoEval = FLAGS.eval_fnglob_or_rate == 0
#-------------------------------------------------------------------------------
BATCH_SIZE = FLAGS.batch_size
NUM_GPUS = FLAGS.num_gpus
assert(BATCH_SIZE % NUM_GPUS == 0)
DEVICE_BATCH_SIZE = BATCH_SIZE / NUM_GPUS
#FLAGS.num_point = Net_Provider.global_num_point
BASE_LEARNING_RATE = FLAGS.learning_rate
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_RATE = FLAGS.decay_rate

# ------------------------------------------------------------------------------
# Load Data
FLAGS.all_fn_globs = FLAGS.all_fn_globs.split(',')
net_configs = {}
net_configs['loss_weight'] = FLAGS.loss_weight
net_provider = Net_Provider(
                            net_configs=net_configs,
                            dataset_name=FLAGS.dataset_name,
                            all_filename_glob=FLAGS.all_fn_globs,
                            eval_fnglob_or_rate=FLAGS.eval_fnglob_or_rate,
                            bxmh5_folder_name = FLAGS.bxmh5_folder_name,
                            only_evaluate = FLAGS.only_evaluate,
                            feed_data_elements=Feed_Data_Elements,
                            feed_label_elements=Feed_Label_Elements)

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

DECAY_STEP = FLAGS.decay_epoch_step * net_provider.train_num_blocks
if TRAIN_FILE_N < 2 or FLAGS.only_evaluate:
    FLAGS.multip_feed = False
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
    if type(FLAGS.eval_fnglob_or_rate)==float and FLAGS.eval_fnglob_or_rate<0:
        log_name = 'log_test_of_train_data.txt'
else:
    MAX_EPOCH = FLAGS.max_epoch
    gsbb_config = net_provider.gsbb_config
    if not FLAGS.finetune:
        log_name = 'log_train.txt'
        nwl_str = '-'+FLAGS.loss_weight + 'lw'
        if FLAGS.inkp_max - FLAGS.inkp_min >0:
            input_dropout_str = '-idp'+str(int((FLAGS.inkp_max - FLAGS.inkp_min)*10))
        else:
            input_dropout_str = ''
        FLAGS.log_dir = FLAGS.log_dir+'-model_'+FLAGS.modelf_nein+nwl_str+input_dropout_str+'-gsbb_'+gsbb_config+'-bs'+str(BATCH_SIZE)+'-'+ \
                        'lr'+str(int(FLAGS.learning_rate*1000))+'-ds_'+str(FLAGS.decay_epoch_step)+'-' + 'Sf_'+ FLAGS.ShuffleFlag + '-'+\
                        FLAGS.feed_data_elements+'-'+str(NUM_POINT)+'-'+FLAGS.dataset_name[0:3]+'_'+str(net_provider.train_num_blocks)
    else:
        log_name = 'log_ft_%d.txt'%(FLAGS.model_epoch)

LOG_DIR = os.path.join(ROOT_DIR,'train_res/semseg_result/'+FLAGS.log_dir)
MODEL_PATH = os.path.join(LOG_DIR,'model.ckpt-'+str(FLAGS.model_epoch))
LOG_DIR_FUSION = os.path.join(ROOT_DIR,'train_res/semseg_result/fusion_log.txt')
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
os.system('cp %s/models/pointnet2_sem_seg.py %s' % (ROOT_DIR,LOG_DIR)) # bkp of model def
os.system('cp %s/train_semseg_sorted.py %s' % (BASE_DIR,LOG_DIR)) # bkp of train procedure
log_fn = os.path.join(LOG_DIR, log_name)
if FLAGS.finetune:
    assert os.path.exists( MODEL_PATH+'.meta' ),"Finetune, but model mote exists: %s"%(MODEL_PATH+'.meta')
if FLAGS.only_evaluate:
    assert os.path.exists( MODEL_PATH+'.meta' ),"Only evaluate, but model mote exists: %s"%(MODEL_PATH+'.meta')
LOG_FOUT = open( log_fn, 'w')
LOG_FOUT_FUSION = open(LOG_DIR_FUSION, 'a')
LOG_FOUT.write(str(FLAGS)+'\n\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.7
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# ------------------------------------------------------------------------------
BLOCK_SAMPLE = net_provider.block_sample
if  DEBUG_SMALLDATA:
    LIMIT_MAX_NUM_BATCHES = {}
    LIMIT_MAX_NUM_BATCHES['train'] = 60
    LIMIT_MAX_NUM_BATCHES['test'] = 60

START_TIME = time.time()

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
# log key parameters
log_string('\n\nkey parameters:')
#log_string( 'data: %s'%(FLAGS.feed_data_elements) )
log_string( 'model: %s'%(FLAGS.modelf_nein) )
log_string( 'sampling & grouping: %s'%(FLAGS.bxmh5_folder_name) )
log_string( 'batch size: %d'%(BATCH_SIZE) )
log_string( 'learning rate: %f'%(FLAGS.learning_rate) )
log_string( 'decay_epoch_step: %d'%(FLAGS.decay_epoch_step) )
if FLAGS.inkp_max - FLAGS.inkp_min >0:
    log_string( 'input dropout: %f - %f'%( FLAGS.inkp_min, FLAGS.inkp_max ) )
else:
    log_string('input dropout: No')

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


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  From tensorflow tutorial: cifar10/cifar10_multi_gpu_train.py
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    #for g, _ in grad_and_vars:
    for g, v in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def train_eval(train_feed_buf_q, train_multi_feed_flags, eval_feed_buf_q, eval_multi_feed_flags, lock):
    with tf.Graph().as_default():
        with tf.device('/cpu:0'):
            #pointclouds_pl, labels_pl,smpws_pl = placeholder_inputs(BATCH_SIZE,BLOCK_SAMPLE,NUM_DATA_ELES,NUM_LABEL_ELES)
            sgf_configs = {}
            sgf_configs['mean_grouping_position'] = FLAGS.group_pos == 'mean' # if not ture, use block center
            sgf_configs['flatten_bm_extract_idx'] = net_provider.flatten_bidxmaps_extract_idx
            sgf_configs['sub_block_stride_candis'] = net_provider.gsbb_load.sub_block_stride_candis
            sgf_configs['sub_block_step_candis'] = net_provider.gsbb_load.sub_block_step_candis
            sgf_configs['sg_bm_extract_idx'] = net_provider.sg_bidxmaps_extract_idx
            sgf_configs['sg_bidxmaps_shape'] = net_provider.sg_bidxmaps_shape
            sgf_configs['flatten_bidxmaps_shape'] = net_provider.flatten_bidxmaps_shape

            pointclouds_pl, labels_pl, smpws_pl,  sg_bidxmaps_pl, flatten_bidxmaps_pl, fbmap_neighbor_dis_pl = placeholder_inputs(BATCH_SIZE,BLOCK_SAMPLE,
                                        NUM_DATA_ELES,NUM_LABEL_ELES, sgf_configs )
            category_labels_pl = labels_pl[...,CATEGORY_LABEL_IDX]
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=global_step parameter to minimize.
            # That tells the optimizer to helpfully increment the 'global_step' parameter for you every time it trains.
            global_step = tf.get_variable('global_step',shape=(),dtype=tf.int32,initializer=tf.zeros_initializer(),trainable=False)
            bn_decay = get_bn_decay(global_step)
            tf.summary.scalar('bn_decay', bn_decay)

            # input drop out
            if FLAGS.inkp_min >= FLAGS.inkp_max:
                input_drop_mask = tf.zeros([])
                print('no input dropout')
            else:
                cas0_point_num = pointclouds_pl.get_shape()[1].value
                input_drop_mask = tf.ones( [DEVICE_BATCH_SIZE, cas0_point_num, 1], tf.float32 )
            input_keep_prob = tf.random_uniform( shape=[], minval=FLAGS.inkp_min, maxval=FLAGS.inkp_max )
            input_drop_mask = tf_util.dropout( input_drop_mask, is_training_pl, 'input_drop_mask', keep_prob = input_keep_prob ) # input_drop_mask/cond/Merge:0

            # Get training operator
            learning_rate = get_learning_rate(global_step)
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('global_step', global_step)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)

            #------------------------------------------
            # Get model and loss on multiple GPUS
            #------------------------------------------
            get_model( FLAGS.modelf_nein, pointclouds_pl, is_training_pl, NUM_CLASSES, sg_bidxmaps_pl,
                       flatten_bidxmaps_pl, fbmap_neighbor_dis_pl, sgf_configs, bn_decay=bn_decay, IsDebug=IS_GEN_PLY)

            tower_grads = []
            pred_gpu = []
            total_loss_gpu = []
            debugs = []
            for gi in range(FLAGS.num_gpus):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    with tf.device('/gpu:%d'%(gi)), tf.name_scope('gpu_%d'%(gi)) as scope:
                        # Evenly split input data to each GPU
                        pc_device = tf.slice(pointclouds_pl, [gi*DEVICE_BATCH_SIZE,0,0], [DEVICE_BATCH_SIZE,-1,-1])
                        label_device = tf.slice(labels_pl, [gi*DEVICE_BATCH_SIZE,0,0], [DEVICE_BATCH_SIZE,-1,-1])
                        smpws_device = tf.slice(smpws_pl, [gi*DEVICE_BATCH_SIZE,0,0], [DEVICE_BATCH_SIZE,-1,-1])
                        sg_bidxmaps_device = tf.slice(sg_bidxmaps_pl, [gi*DEVICE_BATCH_SIZE,0,0], [DEVICE_BATCH_SIZE,-1,-1])
                        flatten_bidxmaps_device = tf.slice(flatten_bidxmaps_pl, [gi*DEVICE_BATCH_SIZE,0,0,0], [DEVICE_BATCH_SIZE,-1,-1,-1])
                        fbmap_neighbor_dis_device = tf.slice(fbmap_neighbor_dis_pl, [gi*DEVICE_BATCH_SIZE,0,0,0], [DEVICE_BATCH_SIZE,-1,-1,-1])

                        pred, end_points, debug = get_model(FLAGS.modelf_nein, pc_device, is_training_pl, NUM_CLASSES, sg_bidxmaps_device,
                                                            flatten_bidxmaps_device, fbmap_neighbor_dis_device, sgf_configs, bn_decay=bn_decay, IsDebug=IS_GEN_PLY )

                        get_loss(pred, label_device, smpws_device, LABEL_ELE_IDXS)
                        losses = tf.get_collection('losses', scope)
                        total_loss = tf.add_n(losses, name='total_loss')
                        for l in losses + [total_loss]:
                            tf.summary.scalar(l.op.name, l)

                        grads = optimizer.compute_gradients(total_loss)
                        tower_grads.append(grads)

                        pred_gpu.append(pred)
                        total_loss_gpu.append(total_loss)
                        debugs.append(debug)

            # Merge pred and losses from multiple GPUs
            pred = tf.concat(pred_gpu, 0)
            total_loss = tf.reduce_mean(total_loss_gpu)

            # Get training operator
            grads = average_gradients(tower_grads)
            train_op = optimizer.apply_gradients(grads, global_step=global_step)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(category_labels_pl))
            accuracy_block = tf.reduce_mean(tf.cast(correct, tf.float32),axis=1)
            accuracy = tf.reduce_mean( accuracy_block )
            tf.summary.scalar('accuracy', accuracy)

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(max_to_keep=10)

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
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': global_step,
               'accuracy_block':accuracy_block }
        ops['pointclouds_pl'] = pointclouds_pl
        ops['labels_pl'] = labels_pl
        ops['smpws_pl'] = smpws_pl
        ops['sg_bidxmaps_pl'] = sg_bidxmaps_pl
        ops['flatten_bidxmaps_pl'] = flatten_bidxmaps_pl
        ops['fbmap_neighbor_dis_pl'] = fbmap_neighbor_dis_pl
        if DEBUG_TMP:
            ops['input_keep_prob'] = input_keep_prob

        if 'l_xyz' in debugs[0]:
            ops['l_xyz'] = [ tf.concat( [ debugs[gi]['l_xyz'][li] for gi in range(FLAGS.num_gpus) ], axis=0 )  for li in range(len(debugs[0]['l_xyz'])) ]
        if 'grouped_xyz' in debugs[0]:
            ops['grouped_xyz'] = [ tf.concat( [ debugs[gi]['grouped_xyz'][li] for gi in range(FLAGS.num_gpus) ], axis=0 )  for li in range(len(debugs[0]['grouped_xyz'])) ]
            ops['flat_xyz'] = [ tf.concat( [ debugs[gi]['flat_xyz'][li] for gi in range(FLAGS.num_gpus) ], axis=0 )  for li in range(len(debugs[0]['flat_xyz'])) ]
            ops['flatten_bidxmap'] = [ tf.concat( [ debugs[gi]['flatten_bidxmap'][li] for gi in range(FLAGS.num_gpus) ], axis=0 )  for li in range(len(debugs[0]['flatten_bidxmap'])) ]

        from tensorflow.contrib.memory_stats.ops import gen_memory_stats_ops
        max_memory_usage = gen_memory_stats_ops.max_bytes_in_use()
        ops['max_memory_usage'] = max_memory_usage

        if FLAGS.finetune:
            saver.restore(sess,MODEL_PATH)
            log_string('finetune, restored model from: \n\t%s'%MODEL_PATH)

        log_string(net_provider.data_summary_str)


        epoch_start = 0
        if FLAGS.finetune or FLAGS.only_evaluate:
            epoch_start+=(FLAGS.model_epoch+1)
        for epoch in range(epoch_start,epoch_start+MAX_EPOCH):
            log_string('**** EPOCH %03d ****   %s' % ( epoch, strftime("%Y-%m-%d %H:%M:%S", gmtime()) ))
            log_string('learning_rate: %f,   IsShuffleIdx: %s'%( sess.run(learning_rate), get_shuffle_flag(epoch) ))
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
                if (epoch > 0 and epoch % 10 == 0) or epoch == MAX_EPOCH-1+epoch_start:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"),global_step=epoch)
                    log_string("Model saved in file: %s" % os.path.basename(save_path))

            if epoch == MAX_EPOCH -1:
                LOG_FOUT_FUSION.write( str(FLAGS)+'\n\n'+train_log_str+'\n'+eval_log_str+'\n\n' )

            #print('train eval finish epoch %d / %d'%(epoch,epoch_start+MAX_EPOCH-1))

def add_log(tot,epoch,batch_idx,loss_batch,t_batch_ls, c_TP_FN_FP = None,numpoint_block=None,all_accuracy=None):
    if type(all_accuracy) != type(None):
        all_accuracy = all_accuracy[0:batch_idx+1]
        ave_block_acc = all_accuracy.mean()
        std_block_acc = all_accuracy.std()
        #cur_batch_acc = all_accuracy[-1,:].mean()
        block_acc_histg = np.histogram( all_accuracy, bins=np.arange(0,1.2,0.1) )[0].astype(np.float32) / all_accuracy.size
    else:
        ave_block_acc, std_block_acc, block_acc_histg, class_acc_str, ave_acc_str = EvaluationMetrics.get_class_accuracy(
                                    c_TP_FN_FP,numpoint_block )
    log_str = '\n'
    if len(t_batch_ls)>0:
        t_per_batch = np.mean(np.concatenate(t_batch_ls,axis=1),axis=1)
        t_per_block = t_per_batch / BATCH_SIZE
        t_per_block_str = np.array2string(t_per_block*1000,formatter={'float_kind':lambda x: "%0.1f"%x})
    else:
        t_per_block_str = "no-t"
    log_str += '%s[%d-%d] t(d,c):%s loss: %0.3f acc: %0.3f-%0.3f' % \
            ( tot,epoch,batch_idx,t_per_block_str,loss_batch,ave_block_acc, std_block_acc )
    log_str += ' acc histgram: %s'%( np.array2string( block_acc_histg,precision=3 ) )
    SimpleFlag = 2
    if type(all_accuracy) == type(None):
        if SimpleFlag >0:
            log_str += '\n' + ave_acc_str
        if  SimpleFlag >1:
            log_str += '\n' + class_acc_str
    log_string(log_str)
    return log_str

def is_complex_log( epoch, batch_idx, num_batches ):
    log_complex = FLAGS.only_evaluate or (epoch % 20 ==0 and epoch>0) and (batch_idx%5 == 0 or batch_idx==num_batches-1)
    return log_complex

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

    num_log_batch = 0
    loss_sum = 0.0
    all_accuracy = np.zeros(shape=(num_batches,BATCH_SIZE),dtype=np.float32)
    c_TP_FN_FP = np.zeros(shape=(num_batches,BATCH_SIZE,NUM_CLASSES,3))

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
            IsShuffleIdx = ( epoch%3 == 0 and FLAGS.ShuffleFlag=='M' ) or FLAGS.ShuffleFlag=='Y'
            cur_data,cur_label,cur_smp_weights,cur_sg_bidxmaps,cur_flatten_bidxmaps, cur_fmap_neighbor_idis, fid_start_end, cur_xyz_mid = net_provider.get_train_batch(start_idx,end_idx,IsShuffleIdx)
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
            cur_data,cur_label,cur_smp_weights, cur_sg_bidxmaps, cur_flatten_bidxmaps, cur_fmap_neighbor_idis,  batch_idx_buf,epoch_buf = train_feed_buf_q.get()
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
        feed_dict[ops['fbmap_neighbor_dis_pl']] = cur_fmap_neighbor_idis

        summary, step, _, loss_val, pred_val, accuracy_batch, max_memory_usage = sess.run( [ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred'], ops['accuracy_block'],ops['max_memory_usage']],
                                    feed_dict=feed_dict )
        t2 = time.time()

        #gen_ply_batch( batch_idx, epoch, sess, ops, feed_dict, cur_label, pred_val, cur_data, accuracy_batch )

        loss_sum += loss_val
        all_accuracy[batch_idx,:] = accuracy_batch
        if ISSUMMARY: train_writer.add_summary(summary, step)
        if is_complex_log(epoch, batch_idx, num_batches):
            pred_logits = np.argmax(pred_val, 2)
            c_TP_FN_FP[num_log_batch,:] = EvaluationMetrics.get_TP_FN_FP(NUM_CLASSES,pred_logits,cur_label[...,CATEGORY_LABEL_IDX])
            num_log_batch += 1
        t_batch_ls.append( np.reshape(np.array([t1-t0, t2-t1, time.time() - t2]),(3,1)) )
        if  batch_idx == num_batches-1 or  (epoch == 0 and batch_idx % 20 ==0) or (batch_idx%20==0):
            train_logstr = add_log('train',epoch,batch_idx,loss_sum/(batch_idx+1),t_batch_ls,all_accuracy = all_accuracy)
            if is_complex_log(epoch, batch_idx, num_batches):
                train_logstr = add_log('train',epoch,batch_idx,loss_sum/(batch_idx+1),t_batch_ls,c_TP_FN_FP = c_TP_FN_FP[0:num_log_batch,:], numpoint_block=NUM_POINT )

        if epoch==0 and batch_idx == 1:
            log_string( 'memory usage: %0.3f G'%(1.0*max_memory_usage/1e9))
            os.system('nvidia-smi')

    print('train epoch %d finished, batch_idx=%d'%(epoch,batch_idx))
    return train_logstr

def gen_ply_batch( batch_idx, epoch, sess, ops, feed_dict, cur_xyz_mid, cur_label, pred_val, cur_data, accuracy_batch, fid_start_end ):
        '''
        lxyz0 == rawdata -> gpxyz0 ->(mean) -> lxyz1 -> gpxyz1 -> lxyz2 -> gpxyz3 -> lxyz4
        gpxyz3 -> flat0,   gpxyz2 -> flat1,  gpxyz1 -> flat2,  gpxyz0 -> flat3
        '''

        show_flag = 'least acc'
        #show_flag = 'all block'

        only_global = True
        if fid_start_end.shape[0] > 1:
            print('batch %d involves more than one file, skip generating ply'%(batch_idx))
            return
        fid_start_end = fid_start_end[0]
        fn = net_provider.get_fn_from_fid( fid_start_end[0] )
        fn_base = os.path.splitext(os.path.basename(fn))[0]

        if show_flag == 'all block':
            bs = bn = BATCH_SIZE
            b0_ls = [0]
        else:
            # bn: how many batches to show
            bn = min(3, BATCH_SIZE)
            bn = min(BATCH_SIZE, BATCH_SIZE)
            bs = 1

            acc_sort = np.argsort( accuracy_batch )
            if show_flag == 'best acc':
                b0_ls = acc_sort[ len(acc_sort)-bn:len(acc_sort) ]
            if show_flag == 'least acc':
                b0_ls = acc_sort[0:bn]

        def plyfn(name_meta, b0, b1):
            acc_str = '%0.3f'%(accuracy_batch[b0:b1].mean())
            acc_str = acc_str[2:len(acc_str)]
            ply_folder = LOG_DIR + '/ply_%s/%s_%d_%d_a0d%s/'%(show_flag, fn_base, fid_start_end[1]+b0, fid_start_end[1]+b1, acc_str )
            if not os.path.exists(ply_folder):
                os.makedirs( ply_folder )
            return ply_folder + name_meta + '_'

        color_flags = ['gt_color']
        cur_xyz_mid = np.expand_dims( cur_xyz_mid, 1 )
        #gen_ply( plyfn(rawdata), cur_data[b0:b1,:,0:3], accuracy_batch,  color_flags,  cur_label[b0:b1], np.argmax(pred_val,2)[b0:b1], cur_data[b0:b1] )
        #pl_display, = sess.run( [ops['pointclouds_pl']], feed_dict=feed_dict )

        if not only_global:
            for lk in range( len(ops['grouped_xyz']) ):
                grouped_xyz, flat_xyz, flatten_bidxmap = sess.run( [ops['grouped_xyz'][lk], ops['flat_xyz'][lk], ops['flatten_bidxmap'][lk] ], feed_dict=feed_dict )
                if 'xyz_midnorm_block' in Feed_Data_Elements:
                    grouped_xyz[...,DATA_ELE_IDXS['xyz_midnorm_block']] += np.expand_dims( cur_xyz_mid,1 )
                    flat_xyz[...,DATA_ELE_IDXS['xyz_midnorm_block']] += cur_xyz_mid
                color_flags = ['no_color']
                for b0 in b0_ls:
                    b1 = b0 + bs
                    gen_ply( plyfn('gpxyz'+str(lk),b0,b1), grouped_xyz[b0:b1,...], accuracy_batch,  color_flags)
                    gen_ply( plyfn('flatxyz'+str(lk),b0,b1), flat_xyz[b0:b1,...], accuracy_batch,  color_flags )
                missed_b_num = np.sum( flatten_bidxmap[...,0,1] < 0 )
                print('missed_b_num:', missed_b_num)

        for lk in range( len(ops['l_xyz']) ):
            pl_display, = sess.run( [ops['l_xyz'][lk]], feed_dict=feed_dict )
            if 'xyz_midnorm_block' in Feed_Data_Elements:
                pl_display[...,DATA_ELE_IDXS['xyz_midnorm_block']] += cur_xyz_mid
            if lk == 0:
                color_flags = ['gt_color','raw_color']
                if 'xyz_midnorm_block' in Feed_Data_Elements:
                    cur_data[...,DATA_ELE_IDXS['xyz_midnorm_block']] += cur_xyz_mid
                for b0 in b0_ls:
                    b1 = b0 + bs
                    gen_ply( plyfn('lxyz0',b0,b1), pl_display[b0:b1,...], accuracy_batch,  color_flags,  cur_label[b0:b1,...], np.argmax(pred_val,2)[b0:b1], cur_data[b0:b1,...] )
            else:
                if not only_global:
                    color_flags = ['no_color']
                    for b0 in b0_ls:
                        b1 = b0 + bs
                        gen_ply( plyfn('lxyz'+str(lk),b0,b1), pl_display[b0:b1,...], accuracy_batch,  color_flags )
        print('sorted acc:\n', np.sort(accuracy_batch))

        if Is_REPORT_PRED:
            pred_fn = plyfn('',0,BATCH_SIZE)+'pred_log.txt'
            EvaluationMetrics.report_pred( pred_fn, pred_val, cur_label[...,CATEGORY_LABEL_IDX], 'matterport3d' )

def gen_ply( ply_fn, pl_display, accuracy_batch, color_flags = ['gt_color'], cur_label=None, pred_val=None, raw_data=None):
    cut_threshold=[1,1,0.85]
    if 'xyz' in Feed_Data_Elements:
        position = 'xyz'
    elif 'xyz_midnorm_block' in Feed_Data_Elements:
        position = 'xyz_midnorm_block'

    if 'no_color' in color_flags:
        cur_xyz = pl_display[...,DATA_ELE_IDXS[position]]
        create_ply_matterport( cur_xyz, ply_fn + 'nocolor.ply', cut_threshold = cut_threshold )

    if 'gt_color' in color_flags:
        cur_xyz = pl_display[...,DATA_ELE_IDXS[position]]
        cur_label_category = cur_label[...,CATEGORY_LABEL_IDX]
        create_ply_matterport( cur_xyz, ply_fn + 'gtcolor.ply', cur_label_category , cut_threshold = cut_threshold )
        create_ply_matterport( cur_xyz, ply_fn + 'predcolor.ply', pred_val, cut_threshold = cut_threshold )
        err_idxs = cur_label_category != pred_val
        create_ply_matterport( cur_xyz[err_idxs], ply_fn + 'err_gtcolor.ply', label = cur_label_category[err_idxs], cut_threshold = [1,1,1] )
        correct_idxs = cur_label_category == pred_val
        create_ply_matterport( cur_xyz[correct_idxs], ply_fn + 'crt_gtcolor.ply', label = cur_label_category[correct_idxs], cut_threshold = [1,1,1] )

    if 'raw_color' in color_flags and 'color_1norm' in DATA_ELE_IDXS:
        cur_xyz_color = raw_data[...,DATA_ELE_IDXS[position]+DATA_ELE_IDXS['color_1norm']]
        cur_xyz_color[...,[3,4,5]] *= 255
        create_ply_matterport( cur_xyz_color, ply_fn + 'rawcolor.ply', cut_threshold = cut_threshold )

def limit_eval_num_batches(epoch,num_batches):
    if epoch%5 != 0:
        num_batches = min(num_batches,31)
    return num_batches

def eval_one_epoch(sess, ops, test_writer, epoch, eval_feed_buf_q, eval_multi_feed_flags, lock):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    num_log_batch = 0
    loss_sum = 0.0

    log_string('----')
    num_blocks = net_provider.eval_num_blocks
    if num_blocks != None:
        num_batches = num_blocks // BATCH_SIZE
        #num_batches = limit_eval_num_batches(epoch,num_batches)
        if DEBUG_SMALLDATA: num_batches = min(num_batches,LIMIT_MAX_NUM_BATCHES['test'])
        all_accuracy = np.zeros(shape=(num_batches,BATCH_SIZE),dtype=np.float32)
        c_TP_FN_FP = np.zeros(shape=(num_batches,BATCH_SIZE,NUM_CLASSES,3))
        if num_batches == 0:
            print('\ntest num_blocks=%d  BATCH_SIZE=%d  num_batches=%d'%(num_blocks,BATCH_SIZE,num_batches))
            return ''
    else:
        assert False
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
            cur_data,cur_label,cur_smp_weights,cur_sg_bidxmaps,cur_flatten_bidxmaps, cur_fmap_neighbor_idis, fid_start_end, cur_xyz_mid  = net_provider.get_eval_batch(start_idx,end_idx,False)
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

            cur_data,cur_label,cur_smp_weights, cur_sg_bidxmaps, cur_flatten_bidxmaps, cur_fmap_neighbor_idis, batch_idx_buf,epoch_buf  = eval_feed_buf_q.get()
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
        feed_dict[ops['fbmap_neighbor_dis_pl']] = cur_fmap_neighbor_idis

        summary, step, loss_val, pred_val,accuracy_batch = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred'],ops['accuracy_block']],
                                      feed_dict=feed_dict)

        if ISSUMMARY and  test_writer != None:
            test_writer.add_summary(summary, step)
        t2 = time.time()

        all_accuracy[batch_idx,:] = accuracy_batch
        loss_sum += loss_val
        if is_complex_log(epoch, batch_idx, num_batches):
            pred_logits = np.argmax(pred_val, 2)
            c_TP_FN_FP[num_log_batch,:] = EvaluationMetrics.get_TP_FN_FP(NUM_CLASSES,pred_logits,cur_label[...,CATEGORY_LABEL_IDX])
            num_log_batch += 1
        t_batch_ls.append( np.reshape(np.array([t1-t0, t2-t1, time.time() - t2]),(3,1)) )
        if batch_idx == num_batches-1 or (batch_idx%20==0):
            eval_logstr = add_log('eval',epoch,batch_idx,loss_sum/(batch_idx+1),t_batch_ls,all_accuracy = all_accuracy )
            if is_complex_log(epoch, batch_idx, num_batches):
                #net_provider.set_pred_label_batch(pred_val,start_idx,end_idx)
                eval_logstr = add_log('eval',epoch,batch_idx,loss_sum/(batch_idx+1),t_batch_ls,c_TP_FN_FP = c_TP_FN_FP[0:num_log_batch,:], numpoint_block=NUM_POINT)

        if IS_GEN_PLY and not FLAGS.multip_feed and batch_idx==0:
            gen_ply_batch( batch_idx, epoch, sess, ops, feed_dict, cur_xyz_mid, cur_label, pred_val, cur_data, accuracy_batch, fid_start_end )
            return
    return eval_logstr


def get_shuffle_flag(epoch):
    IsShuffleIdx = ( epoch%3 == 0 and FLAGS.ShuffleFlag=='M' ) or FLAGS.ShuffleFlag=='Y'
    return IsShuffleIdx

def add_feed_buf(train_or_test,feed_buf_q, cpu_id, file_id_start, file_id_end, multi_feed_flags, lock, limit_max_train_num_batches=None):
    with tf.device('/cpu:%d'%(cpu_id)):
        max_buf_size = 3
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
            IsShuffleIdx = get_shuffle_flag(epoch)
            #IsShuffleIdx = False
            if cpu_id==0:
                if IsShuffleIdx: net_provider.update_train_eval_shuffled_idx()

            batch_idx = -1 + batch_idx_start
            while (batch_idx < num_batches-1 + batch_idx_start) or (num_batches==None):
                if feed_buf_q.qsize() < max_buf_size:
                    batch_idx += 1
                    block_start_idx = batch_idx * BATCH_SIZE
                    block_end_idx = (batch_idx+1) * BATCH_SIZE
                    if train_or_test == 'train':
                        cur_data,cur_label,cur_smp_weights,cur_sg_bidxmaps,cur_flatten_bidxmaps, cur_fmap_neighbor_idis, fid_start_end, cur_xyz_mid  = net_provider.get_train_batch(block_start_idx,block_end_idx,IsShuffleIdx)
                    elif train_or_test == 'test':
                        cur_data,cur_label,cur_smp_weights,cur_sg_bidxmaps,cur_flatten_bidxmaps, cur_fmap_neighbor_idis, fid_start_end, cur_xyz_mid  = net_provider.get_eval_batch(block_start_idx,block_end_idx,False)
                    feed_buf_q.put( [cur_data,cur_label,cur_smp_weights, cur_sg_bidxmaps, cur_flatten_bidxmaps, cur_fmap_neighbor_idis, batch_idx,epoch] )
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
