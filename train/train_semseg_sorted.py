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
from ply_util import create_ply_matterport

ISSUMMARY = False
TMP_DEBUG = False
IS_GEN_PLY = False
if IS_GEN_PLY:
    IsShuffleIdx = False
else:
    IsShuffleIdx = True
LOG_TYPE = 'simple'

parser = argparse.ArgumentParser()
parser.add_argument('--model_flag', default='3A', help='model flag')
parser.add_argument('--dataset_name', default='matterport3d', help='dataset_name: scannet, stanford_indoor,matterport3d')
parser.add_argument('--datafeed_type', default='SortedH5f', help='SortedH5f or Normed_H5f or Pr_NormedH5f')
parser.add_argument('--all_fn_globs', type=str,default='all_merged_nf5/stride_0d1_step_0d1_pyramid-1d6_2-512_256_64-128_12_6-0d2_0d6_1d2',\
                    help='The file name glob for both training and evaluation')
parser.add_argument('--feed_data_elements', default='xyz_midnorm', help='xyz_1norm,xyz_midnorm,color_1norm')
parser.add_argument('--feed_label_elements', default='label_category', help='label_category,label_instance')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 24]')
parser.add_argument('--eval_fnglob_or_rate',  default='test', help='file name str glob or file number rate: scan1*.nh5 0.2')
parser.add_argument('--num_point', type=int, default=-1, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')

parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
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

if FLAGS.datafeed_type == 'Normed_H5f':
    from pointnet2_sem_seg import  placeholder_inputs,get_model,get_loss
elif FLAGS.datafeed_type == 'Pr_Normed_H5f':
    from pointnet2_sem_seg_pyramid_feed import  placeholder_inputs,get_model,get_loss

BATCH_SIZE = FLAGS.batch_size
if FLAGS.datafeed_type == 'Pr_Normed_H5f':
    FLAGS.num_point = Net_Provider.global_num_point
NUM_POINT = FLAGS.num_point
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

try:
    FLAGS.eval_fnglob_or_rate = float(FLAGS.eval_fnglob_or_rate)
    log_eval_fn_glob = ''
    print('FLAGS.eval_fnglob_or_rate is eval file number rate')
except:
    log_eval_fn_glob = FLAGS.eval_fnglob_or_rate.split('*')[0]
    print('FLAGS.eval_fnglob_or_rate is eval name glob. log_eval_fn_glob:%s'%(log_eval_fn_glob))

if FLAGS.only_evaluate:
    MAX_EPOCH = 1
    log_name = 'log_Test.txt'
else:
    MAX_EPOCH = FLAGS.max_epoch
    log_name = 'log_Train.txt'
    gsbb_config = Net_Provider.gsbb_config
    FLAGS.log_dir = FLAGS.log_dir+'-gsbb_'+gsbb_config+'-B'+str(BATCH_SIZE)+'-'+\
                    FLAGS.feed_data_elements+'-'+str(FLAGS.num_point)+'-'+FLAGS.dataset_name+'-eval_'+log_eval_fn_glob
FLAGS.feed_data_elements = FLAGS.feed_data_elements.split(',')
FLAGS.feed_label_elements = FLAGS.feed_label_elements.split(',')

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
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# Load Data
FLAGS.all_fn_globs = FLAGS.all_fn_globs.split(',')
net_provider = Net_Provider(InputType=FLAGS.datafeed_type,
                            dataset_name=FLAGS.dataset_name,
                            all_filename_glob=FLAGS.all_fn_globs,
                            eval_fnglob_or_rate=FLAGS.eval_fnglob_or_rate,
                            only_evaluate = FLAGS.only_evaluate,
                            num_point_block = FLAGS.num_point,
                            feed_data_elements=FLAGS.feed_data_elements,
                            feed_label_elements=FLAGS.feed_label_elements)
NUM_DATA_ELES = net_provider.data_num_eles
NUM_CLASSES = net_provider.num_classes
NUM_LABEL_ELES = net_provider.label_num_eles
LABEL_ELE_IDXS = net_provider.feed_label_ele_idxs
DATA_ELE_IDXS = net_provider.feed_data_ele_idxs
CATEGORY_LABEL_IDX = LABEL_ELE_IDXS['label_category'][0]

BLOCK_SAMPLE = net_provider.block_sample

START_TIME = time.time()



def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train_eval(train_feed_buf_q,eval_feed_buf_q):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            if FLAGS.datafeed_type == 'Normed_H5f':
                pointclouds_pl, labels_pl,smpws_pl = placeholder_inputs(BATCH_SIZE,BLOCK_SAMPLE,NUM_DATA_ELES,NUM_LABEL_ELES)
            elif FLAGS.datafeed_type == 'Pr_Normed_H5f':
                flatten_bm_extract_idx = net_provider.flatten_bidxmaps_extract_idx
                grouped_pointclouds_pl, grouped_labels_pl, grouped_smpws_pl, sg_bidxmaps_pl, flatten_bidxmaps_pl, labels_pl, smpws_pl, debug_pls = placeholder_inputs(BATCH_SIZE,BLOCK_SAMPLE,
                                        NUM_DATA_ELES,NUM_LABEL_ELES,net_provider.sg_bidxmaps_shape,net_provider.flatten_bidxmaps_shape, flatten_bm_extract_idx )
                flat_pointclouds_pl = debug_pls['flat_pointclouds']
            category_labels_pl = labels_pl[...,CATEGORY_LABEL_IDX]
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            if FLAGS.datafeed_type == 'Normed_H5f':
                pred,end_points = get_model( FLAGS.model_flag, pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
                loss = get_loss(pred, labels_pl,smpws_pl)
            elif FLAGS.datafeed_type == 'Pr_Normed_H5f':
                sg_bm_extract_idx = net_provider.sg_bidxmaps_extract_idx
                pred,end_points = get_model( FLAGS.model_flag, grouped_pointclouds_pl, is_training_pl, NUM_CLASSES, sg_bidxmaps_pl, sg_bm_extract_idx, flatten_bidxmaps_pl, flatten_bm_extract_idx,  bn_decay=bn_decay)
                loss = get_loss(pred, labels_pl, smpws_pl, LABEL_ELE_IDXS)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(category_labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

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
               'step': batch,
               'accuracy':accuracy}
        if FLAGS.datafeed_type == 'Normed_H5f':
            ops['pointclouds_pl'] = pointclouds_pl
            ops['labels_pl'] = labels_pl
            ops['smpws_pl'] = smpws_pl
        elif FLAGS.datafeed_type == 'Pr_Normed_H5f':
            ops['grouped_pointclouds_pl'] = grouped_pointclouds_pl
            ops['grouped_labels_pl'] = grouped_labels_pl
            ops['labels_pl'] = labels_pl
            ops['grouped_smpws_pl'] = grouped_smpws_pl
            ops['sg_bidxmaps_pl'] = sg_bidxmaps_pl
            ops['flatten_bidxmaps_pl'] = flatten_bidxmaps_pl
            ops['flat_pointclouds_pl'] = flat_pointclouds_pl

            if ISDEBUG:
                ops['flatten_bidxmaps_pl_0'] = debug_pls['flatten_bidxmaps_pl_0']

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
                train_log_str = train_one_epoch(sess, ops, train_writer,epoch,train_feed_buf_q)
            else:
                train_log_str = ''
                saver.restore(sess,MODEL_PATH)
                log_string('only evaluate, restored model from: \n\t%s'%MODEL_PATH)
            eval_log_str = eval_one_epoch(sess, ops, test_writer,epoch,eval_feed_buf_q)

            # Save the variables to disk.
            if not FLAGS.only_evaluate:
                if (epoch > 0 and epoch % 1 == 0) or epoch == MAX_EPOCH-1:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"),global_step=epoch)
                    log_string("Model saved in file: %s" % os.path.basename(save_path))

            if epoch == MAX_EPOCH -1:
                LOG_FOUT_FUSION.write( str(FLAGS)+'\n\n'+train_log_str+'\n'+eval_log_str+'\n\n' )



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

def train_one_epoch(sess, ops, train_writer,epoch,train_feed_buf_q):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    #log_string('----')
    num_blocks = net_provider.train_num_blocks
    if num_blocks!=None:
        num_batches = num_blocks // BATCH_SIZE
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
    while (batch_idx < num_batches-1) or (num_batches==None):
        t0 = time.time()
        batch_idx += 1
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        if train_feed_buf_q == None:
            cur_data,cur_label,cur_smp_weights,cur_sg_bidxmaps,cur_flatten_bidxmaps = net_provider.get_train_batch(start_idx,end_idx,IsShuffleIdx)
        else:
            if train_feed_buf_q.qsize() == 0:
                print('train_feed_buf_q.qsize == 0')
                break
            cur_data,cur_label,cur_smp_weights, cur_sg_bidxmaps, cur_flatten_bidxmaps,  batch_idx_buf,epoch_buf = train_feed_buf_q.get()
            #assert batch_idx == batch_idx_buf and epoch== epoch_buf

        t1 = time.time()
        if type(cur_data) == type(None):
            break # all data reading finished
        label_category = cur_label[:,:,CATEGORY_LABEL_IDX]
        feed_dict = { ops['is_training_pl']: is_training }
        if FLAGS.datafeed_type == 'Normed_H5f':
            feed_dict[ops['pointclouds_pl']] = cur_data
            feed_dict[ops['labels_pl']] = cur_label
            feed_dict[ops['smpws_pl']] = cur_smp_weights
        elif FLAGS.datafeed_type == 'Pr_Normed_H5f':
            feed_dict[ops['grouped_pointclouds_pl']] = cur_data
            feed_dict[ops['grouped_labels_pl']] = cur_label
            feed_dict[ops['grouped_smpws_pl']] = cur_smp_weights
            feed_dict[ops['sg_bidxmaps_pl']] = cur_sg_bidxmaps
            feed_dict[ops['flatten_bidxmaps_pl']] = cur_flatten_bidxmaps

        #if ISDEBUG:
        #    grouped_labels, flatten_bidxmaps_0 =  sess.run( [ops['grouped_labels_pl'], ops['flatten_bidxmaps_pl_0']] ,
        #                        feed_dict=feed_dict)
        if FLAGS.datafeed_type == 'Pr_Normed_H5f':
            cur_flatten_label, = sess.run( [ops['labels_pl']], feed_dict=feed_dict )
            cur_label = cur_flatten_label
            if IS_GEN_PLY:
                cur_flatten_pointcloud, = sess.run( [ops['flat_pointclouds_pl']], feed_dict=feed_dict )
                color_flag = 'raw_color'
                if color_flag == 'gt_color':
                    cur_xyz = cur_flatten_pointcloud[...,DATA_ELE_IDXS['xyz']]
                    create_ply_matterport( cur_xyz, LOG_DIR+'/train_flat_%d_gtcolor'%(batch_idx)+'.ply', cur_flatten_label[...,CATEGORY_LABEL_IDX] )
                if color_flag == 'raw_color':
                    cur_xyz_color = cur_flatten_pointcloud[...,DATA_ELE_IDXS['xyz']+DATA_ELE_IDXS['color_1norm']]
                    cur_xyz_color[...,[3,4,5]] *= 255
                    create_ply_matterport( cur_xyz_color, LOG_DIR+'/train_flat_%d_rawcolor'%(batch_idx)+'.ply' )
                    cur_xyz_color = cur_data[...,DATA_ELE_IDXS['xyz']+DATA_ELE_IDXS['color_1norm']]
                    cur_xyz_color[...,[3,4,5]] *= 255
                    create_ply_matterport( cur_xyz_color, LOG_DIR+'/train_grouped_%d_rawcolor'%(batch_idx)+'.ply' )
                import pdb; pdb.set_trace()  # XXX BREAKPOINT


        summary, step, _, loss_val, pred_val, accuracy_batch = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred'], ops['accuracy']],
                                    feed_dict=feed_dict)

        loss_sum += loss_val
        accuracy_sum += accuracy_batch
        t_batch_ls.append( np.reshape(np.array([t1-t0,time.time() - t1]),(2,1)) )
        if ISSUMMARY: train_writer.add_summary(summary, step)
        if batch_idx == num_batches-1 or  (epoch == 0 and batch_idx % 20 ==0) or (batch_idx%200==0 and batch_idx>0):
            if LOG_TYPE == 'complex':
                pred_val = np.argmax(pred_val, 2)
                total_seen += (BATCH_SIZE*NUM_POINT)
                c_TP_FN_FP += EvaluationMetrics.get_TP_FN_FP(NUM_CLASSES,pred_val,cur_label[...,CATEGORY_LABEL_IDX])
                train_logstr = add_log('train',epoch,batch_idx,loss_sum/(batch_idx+1),t_batch_ls,c_TP_FN_FP = c_TP_FN_FP,total_seen = total_seen)
            else:
                train_logstr = add_log('train',epoch,batch_idx,loss_sum/(batch_idx+1),t_batch_ls,accuracy = accuracy_sum/(batch_idx+1))
        if batch_idx == 100:
            os.system('nvidia-smi')
    return train_logstr

def limit_eval_num_batches(epoch,num_batches):
    if epoch%5 != 0:
        num_batches = min(num_batches,31)
    return num_batches

def eval_one_epoch(sess, ops, test_writer, epoch,eval_feed_buf_q):
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
        num_batches = limit_eval_num_batches(epoch,num_batches)
        if num_batches == 0:
            print('\ntest num_blocks=%d  BATCH_SIZE=%d  num_batches=%d'%(num_blocks,BATCH_SIZE,num_batches))
            return ''
    else:
        num_batches = None

    eval_logstr = ''
    t_batch_ls = []
    batch_idx = -1

    while (batch_idx < num_batches-1) or (num_batches==None):
        t0 = time.time()
        batch_idx += 1
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        if eval_feed_buf_q == None:
            cur_data,cur_label,cur_smp_weights,cur_sg_bidxmaps,cur_flatten_bidxmaps  = net_provider.get_eval_batch(start_idx,end_idx,IsShuffleIdx)
        else:
            if eval_feed_buf_q.qsize() == 0:
                print('eval_feed_buf_q.qsize == 0')
                break
            cur_data,cur_label,cur_smp_weights, cur_sg_bidxmaps, cur_flatten_bidxmaps, batch_idx_buf,epoch_buf  = eval_feed_buf_q.get()
            #assert batch_idx == batch_idx_buf and epoch== epoch_buf

        t1 = time.time()
        if type(cur_data) == type(None):
            print('batch_idx:%d, get None, reading finished'%(batch_idx))
            break # all data reading finished

        feed_dict = { ops['is_training_pl']: is_training }
        if FLAGS.datafeed_type == 'Normed_H5f':
            feed_dict[ops['pointclouds_pl']] = cur_data
            feed_dict[ops['labels_pl']] = cur_label
            feed_dict[ops['smpws_pl']] = cur_smp_weights
        elif FLAGS.datafeed_type == 'Pr_Normed_H5f':
            feed_dict[ops['grouped_pointclouds_pl']] = cur_data
            feed_dict[ops['grouped_labels_pl']] = cur_label
            feed_dict[ops['grouped_smpws_pl']] = cur_smp_weights
            feed_dict[ops['sg_bidxmaps_pl']] = cur_sg_bidxmaps
            feed_dict[ops['flatten_bidxmaps_pl']] = cur_flatten_bidxmaps

        if FLAGS.datafeed_type == 'Pr_Normed_H5f':
            cur_flatten_label, = sess.run( [ops['labels_pl']], feed_dict=feed_dict )
            cur_label = cur_flatten_label
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

    return eval_logstr


def add_train_feed_buf(train_feed_buf_q):
    with tf.device('/cpu:0'):
        max_buf_size = 20
        num_blocks = net_provider.train_num_blocks
        if num_blocks!=None:
            num_batches = num_blocks // BATCH_SIZE
        else:
            num_batches = None

        epoch_start = 0
        if FLAGS.finetune:
            epoch_start+=(FLAGS.model_epoch+1)
        for epoch in range(epoch_start,epoch_start+MAX_EPOCH):
            net_provider.update_train_eval_shuffled_idx()
            batch_idx = -1
            while (batch_idx < num_batches-1) or (num_batches==None):
                if train_feed_buf_q.qsize() < max_buf_size:
                    batch_idx += 1
                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = (batch_idx+1) * BATCH_SIZE
                    cur_data,cur_label,cur_smp_weights,cur_sg_bidxmaps,cur_flatten_bidxmaps  = net_provider.get_train_batch(start_idx,end_idx)
                    train_feed_buf_q.put( [cur_data,cur_label,cur_smp_weights, cur_sg_bidxmaps, cur_flatten_bidxmaps, batch_idx,epoch] )
                    if type(cur_data) == type(None):
                        print('add_train_feed_buf: get None data from net_provider, all data put finished. epoch= %d, batch_idx= %d'%(epoch,batch_idx))
                        break # all data reading finished
                else:
                    time.sleep(0.1*BATCH_SIZE*max_buf_size/3)
            print('add_train_feed_buf: data reading finished. epoch= %d, batch_idx= %d'%(epoch,batch_idx))

def add_eval_feed_buf(eval_feed_buf_q):
    with tf.device('/cpu:1'):
        max_buf_size = 20
        num_blocks = net_provider.eval_num_blocks
        if num_blocks!=None:
            raw_num_batches = num_blocks // BATCH_SIZE
        else:
            raw_num_batches = None

        epoch_start = 0
        if FLAGS.finetune:
            epoch_start+=(FLAGS.model_epoch+1)
        for epoch in range(epoch_start,epoch_start+MAX_EPOCH):
            batch_idx = -1
            num_batches = limit_eval_num_batches(epoch,raw_num_batches)
            while (batch_idx < num_batches-1) or (num_batches==None):
                if eval_feed_buf_q.qsize() < max_buf_size:
                    batch_idx += 1
                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = (batch_idx+1) * BATCH_SIZE
                    cur_data,cur_label,cur_smp_weights,cur_sg_bidxmaps,cur_flatten_bidxmaps  = net_provider.get_eval_batch(start_idx,end_idx)
                    eval_feed_buf_q.put( [cur_data,cur_label,cur_smp_weights, cur_sg_bidxmaps, cur_flatten_bidxmaps, batch_idx,epoch] )
                    if type(cur_data) == type(None):
                        print('add_eval_feed_buf: get None data from net_provider, all data put finished. epoch= %d, batch_idx= %d'%(epoch,batch_idx))
                        break # all data reading finished
                else:
                    time.sleep(0.1*BATCH_SIZE*max_buf_size/3)
            print('add_eval_feed_buf: data reading finished. epoch= %d, batch_idx= %d'%(epoch,batch_idx))


def main():

    IsFeedData_MultiProcessing = FLAGS.multip_feed and (not FLAGS.auto_break)


    if IsFeedData_MultiProcessing:
        train_feed_buf_q = mp.Queue()
        eval_feed_buf_q = mp.Queue()

        processes = {}
        processes[ 'add_train_buf'] = mp.Process(target=add_train_feed_buf,args=(train_feed_buf_q,))
        processes[ 'add_eval_buf'] = mp.Process(target=add_eval_feed_buf,args=(eval_feed_buf_q,))
        processes[ 'train_eval'] = mp.Process(target=train_eval,args=(train_feed_buf_q,eval_feed_buf_q,))

        for p in processes:
            processes[p].start()
        for p in processes:
            processes[p].join()
    else:
        train_eval(None,None)

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
