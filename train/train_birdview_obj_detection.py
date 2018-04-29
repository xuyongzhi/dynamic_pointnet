#!/usr/bin/env python
# --------------------------------------------------------
# 3D object detection train file
#
# -------------------------------------------------------

import pdb, traceback
import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import time
import os
import sys
import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,'utils_xyz'))
sys.path.append(os.path.join(ROOT_DIR,'models'))
sys.path.append(os.path.join(ROOT_DIR,'config'))
sys.path.append(os.path.join(ROOT_DIR,'utils'))

# from pointnet2_obj_detection_tf4 import  placeholder_inputs,get_model,get_loss
# import get_dataset
# from evaluation import EvaluationMetrics
# from kitti_data_net_provider_2d import kitti_data_net_provider_2d #Normed_H5f,Net_Provider
from block_data_net_provider_kitti_2d import Normed_H5f,Net_Provider_kitti

import tf_util
from config import cfg
import multiprocessing as mp
from bbox_transform_2d import bbox_transform_inv_2d
from nms_2d import nms_2d
from evaluation_2d import evaluation_2d
# to check the memory usage of GPU
from pointnet2_sem_seg_presg_kitti_2d import  placeholder_inputs,get_model,get_loss
from tensorflow.contrib.memory_stats.ops import gen_memory_stats_ops



ISDEBUG = False
ISSUMMARY = True

parser = argparse.ArgumentParser()
parser.add_argument('--modelf_nein', default='3AG_114', help='{model flag}_{neighbor num of cascade 0,0 from 1,and others}')
parser.add_argument('--dataset_name', default='rawh5_kitti_32768', help='rawh5_kitti')
parser.add_argument('--all_fn_globs', type=str,default='Merged_sph5/90000_gs-4_-6d3/', help='The file name glob for both training and evaluation')

parser.add_argument('--bxmh5_folder_name', default='Merged_bxmh5/90000_gs-4_-6d3_fmn6-6400_2400_320_32-32_16_32_48-0d1_0d3_0d9_2d7-0d1_0d2_0d6_1d8-pd3-4C0', help='')
parser.add_argument('--feed_data_elements', default='xyz', help='xyz_1norm_file-xyz_midnorm_block-color_1norm')
parser.add_argument('--feed_label_elements', default='label_category', help='label_category-label_instance')

parser.add_argument('--batch_size', type=int, default= 32, help='Batch Size during training [default: 24]')
parser.add_argument('--eval_fnglob_or_rate',  default='0.2', help='file name str glob or file number rate: scan1*.nh5 0.2')
parser.add_argument('--num_point', type=int, default=2**15, help='Point number [default: 2**15]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')

parser.add_argument('--group_pos',default='mean',help='mean or bc(block center)')


parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate [default: 0.01]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--max_test_file_num', type=int, default=None, help='Which area to use for test, option: 1-6 [default: 6]')

parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')

parser.add_argument('--gpu', type=int, default= 0, help='the gpu index')

parser.add_argument('--substract_center', action='store_true',  help='whethere to substract center point')

parser.add_argument('--only_evaluate',action='store_true',help='do not train')
parser.add_argument('--finetune',action='store_true',help='do not train')
parser.add_argument('--model_epoch', type=int, default=10, help='the epoch of model to be restored')
parser.add_argument('--auto_break',action='store_true',help='If true, auto break when error occurs')
parser.add_argument('--loss_weight', default='E', help='E: Equal, N:Number, C:Center, CN')

parser.add_argument('--inkp_min', type=float, default=1.0, help='random input drop minimum')
parser.add_argument('--inkp_max', type=float, default=1.0, help='random input drop maxmum')



FLAGS = parser.parse_args()

DATASET_NAME = FLAGS.dataset_name
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

IS_GEN_PLY= True and FLAGS.only_evaluate


try:
    FLAGS.eval_fnglob_or_rate = float(FLAGS.eval_fnglob_or_rate)
    log_eval_fn_glob = ''
    print('FLAGS.eval_fnglob_or_rate is eval file number rate')
except:
    log_eval_fn_glob = FLAGS.eval_fnglob_or_rate.split('*')[0]
    print('FLAGS.eval_fnglob_or_rate is eval name glob. log_eval_fn_glob:%s'%(log_eval_fn_glob))

date = datetime.datetime.now().date()
if FLAGS.only_evaluate:
    MAX_EPOCH = 1
    log_name = 'log_Test.txt'
else:
    MAX_EPOCH = FLAGS.max_epoch
    log_name = 'log_Train.txt'
    FLAGS.log_dir = FLAGS.log_dir+'-B'+str(BATCH_SIZE)+'-'+\
                    FLAGS.feed_data_elements+'-'+str(NUM_POINT)+'-'+FLAGS.dataset_name+'-eval_'+log_eval_fn_glob+str(date)
# FLAGS.feed_data_elements = FLAGS.feed_data_elements.split(',')
Feed_Data_Elements  = FLAGS.feed_data_elements.split('-')
Feed_Label_Elements = FLAGS.feed_label_elements.split('-')


LOG_DIR = os.path.join(ROOT_DIR,'train_res/object_detection_result/'+FLAGS.log_dir)
MODEL_PATH = os.path.join(LOG_DIR,'model.ckpt-'+str(FLAGS.model_epoch))
LOG_DIR_FUSION = os.path.join(ROOT_DIR,'train_res/object_detection_result/accuracy_log.txt')
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
#os.system('cp %s/models/pointnet2_obj_detection_tf4.py %s' % (ROOT_DIR,LOG_DIR)) # bkp of model def
os.system('cp %s/config/config.py %s' % (ROOT_DIR,LOG_DIR))
os.system('cp %s/models/pointnet2_sem_seg_presg_kitti_2d.py %s' % (ROOT_DIR,LOG_DIR)) # bkp of model def
os.system('cp %s/train/train_birdview_obj_detection.py %s' % (ROOT_DIR,LOG_DIR)) # bkp of train procedure
os.system('cp %s/utils_xyz/configs_kitti.py %s' % (ROOT_DIR,LOG_DIR)) # bkp of train procedure
os.system('cp %s/utils_xyz/pointnet_blockid_sg_util_kitti.py %s' % (ROOT_DIR,LOG_DIR)) # bkp of train procedure


acc_name = 'accuracy.txt'
if FLAGS.finetune:
    LOG_FOUT = open(os.path.join(LOG_DIR, log_name), 'a')
    LOG_FOUT_FUSION = open(os.path.join(LOG_DIR, acc_name), 'a')
else:
    LOG_FOUT = open(os.path.join(LOG_DIR, log_name), 'w')
    LOG_FOUT_FUSION = open(os.path.join(LOG_DIR, acc_name), 'w')

if FLAGS.finetune:
    assert os.path.exists( MODEL_PATH+'.meta' ),"Finetune, but model mote exists: %s"%(MODEL_PATH+'.meta')
if FLAGS.only_evaluate:
    assert os.path.exists( MODEL_PATH+'.meta' ),"Only evaluate, but model mote exists: %s"%(MODEL_PATH+'.meta')

LOG_FOUT_FUSION.write(str(FLAGS)+'\n\n')
LOG_FOUT.write(str(FLAGS)+'\n\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# Load Data
#data_provider = kitti_data_net_provider_2d(DATASET_NAME,BATCH_SIZE)

net_configs = {}
net_configs['loss_weight'] = FLAGS.loss_weight
ALL_fn_globs = FLAGS.all_fn_globs.split(',')


net_provider = Net_Provider_kitti(
                            net_configs=net_configs,
                            dataset_name=FLAGS.dataset_name,
                            all_filename_glob=ALL_fn_globs,
                            eval_fnglob_or_rate=FLAGS.eval_fnglob_or_rate,
                            bxmh5_folder_name = FLAGS.bxmh5_folder_name,
                            only_evaluate = FLAGS.only_evaluate,
                            feed_data_elements=Feed_Data_Elements,
                            feed_label_elements=Feed_Label_Elements)


NUM_POINT = net_provider.global_num_point
NUM_DATA_ELES = net_provider.data_num_eles
NUM_LABEL_ELES= net_provider.label_num_eles

TRAIN_FILE_N = net_provider.train_file_N
EVAL_FILE_N = net_provider.eval_file_N
MAX_MULTIFEED_NUM = 5

LABEL_ELE_IDXS = net_provider.feed_label_ele_idxs
DATA_ELE_IDXS = net_provider.feed_data_ele_idxs

NUM_CHANNELS = cfg.TRAIN.NUM_CHANNELS  # x, y, z
NUM_CLASSES =  cfg.TRAIN.NUM_CLASSES   # bg(car), fg
NUM_REGRESSION = cfg.TRAIN.NUM_REGRESSION


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
            # pointclouds_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT,NUM_CHANNELS, NUM_REGRESSION)
            sgf_configs = {}
            sgf_configs['mean_grouping_position'] =  FLAGS.group_pos == 'mean' # if not ture, use block center
            sgf_configs['flatten_bm_extract_idx'] = net_provider.flatten_bidxmaps_extract_idx
            sgf_configs['sub_block_stride_candis'] = net_provider.gsbb_load.sub_block_stride_candis
            sgf_configs['sub_block_step_candis'] = net_provider.gsbb_load.sub_block_step_candis
            sgf_configs['sg_bm_extract_idx'] = net_provider.sg_bidxmaps_extract_idx
            sgf_configs['sg_bidxmaps_shape'] = net_provider.sg_bidxmaps_shape
            sgf_configs['flatten_bidxmaps_shape'] = net_provider.flatten_bidxmaps_shape
            sgf_configs['substract_center'] = FLAGS.substract_center

            sgf_configs['Cnn_keep_prob'] = 1

            # pointclouds_pl,  labels_pl, smpws_pl, sg_bidxmaps_pl = placeholder_inputs(BATCH_SIZE, BLOCK_SAMPLE, NUM_DATA_ELES, NUM_LABEL_ELES, sgf_configs['sg_bidxmaps_shape'], NUM_REGRESSION)
            pointclouds_pl,  labels_pl, sg_bidxmaps_pl , sgf_config_pls = placeholder_inputs(BATCH_SIZE, BLOCK_SAMPLE, NUM_DATA_ELES, NUM_LABEL_ELES, sgf_configs['sg_bidxmaps_shape'], NUM_REGRESSION)

            # category_labels_pl = labels_pl[...,CATEGORY_LABEL_IDX]
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)   ## batch has the same concept with global_step
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)


            ## input drop out to use small model learn big data
            if FLAGS.inkp_min >= FLAGS.inkp_max:
                input_drop_mask = tf.zeros([])
                print('no input dropout')
            else:
                cas0_point_num = pointclouds_pl.get_shape()[1].value
                input_drop_mask = tf.ones( [BATCH_SIZE, cas0_point_num, 1], tf.float32 )
            input_keep_prob = tf.random_uniform( shape=[], minval=FLAGS.inkp_min, maxval=FLAGS.inkp_max )
            input_drop_mask = tf_util.dropout( input_drop_mask, is_training_pl, scope='dropout', keep_prob = input_keep_prob, name='input_dropout_mask')

            # Get model and loss
            # pred_class, pred_box, xyz_pl = get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
            # pred, end_points, debug = get_model( FLAGS.modelf_nein, pointclouds_pl, is_training_pl, NUM_CLASSES, sg_bidxmaps_pl,
            #                                        sg_bm_extract_idx, flatten_bidxmaps_pl, fbmap_neighbor_dis_pl, flatten_bm_extract_idx, bn_decay=bn_decay, IsDebug=IS_GEN_PLY)


            pred_class, pred_box, xyz_pl = get_model(FLAGS.modelf_nein, pointclouds_pl, is_training_pl, NUM_CLASSES, sg_bidxmaps_pl,
                                                 sgf_configs, sgf_config_pls, bn_decay=bn_decay, IsDebug=IS_GEN_PLY )


            loss, classification_loss, regression_loss, loss_details, pred_prob, accuracy_classification, recall_classification, num_positive_label = get_loss(BATCH_SIZE, pred_class, pred_box, labels_pl, xyz_pl)
            tf.summary.scalar('loss', loss)

            #correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            #accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            #tf.summary.scalar('accuracy', accuracy)

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
        sess = tf.Session(config=config)

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

        max_memory_usage = gen_memory_stats_ops.max_bytes_in_use()

        # define operations
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'sg_bidxmaps_pl': sg_bidxmaps_pl,
               'is_training_pl': is_training_pl,
               'pred_class': pred_class,
               'pred_box': pred_box,
               'globalb_bottom_center_xyz': sgf_config_pls['globalb_bottom_center_xyz'],
               'xyz_pl': xyz_pl,
               'loss': loss,
               'classification_loss':classification_loss,
               'regression_loss':regression_loss,
               'loss_details':loss_details,
               'accuracy_classification':accuracy_classification,
               'recall_classification':recall_classification,
               'num_positive_label':num_positive_label,
               'pred_prob':pred_prob,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'max_memory_usage': max_memory_usage
               }
        if FLAGS.finetune:
            saver.restore(sess,MODEL_PATH)
            log_string('finetune, restored model from: \n\t%s'%MODEL_PATH)

        log_string(net_provider.data_summary_str)

        if ISDEBUG:
            builder = tf.profiler.ProfileOptionBuilder
            opts = builder(builder.time_and_memory()).order_by('micros').build()
            pctx =  tf.contrib.tfprof.ProfileContext('/tmp/train_dir',
                                    trace_steps=[],
                                    dump_steps=[])
        else:
            opts = None
            pctx = None

        epoch_start = 0
        if FLAGS.finetune:
            epoch_start+=(FLAGS.model_epoch+1)
        for epoch in range(epoch_start,epoch_start+MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            if not FLAGS.only_evaluate:
                train_log_str = train_one_epoch(sess, ops, train_writer,epoch,train_feed_buf_q,pctx,opts)
            else:
                train_log_str = ''
                saver.restore(sess,MODEL_PATH)
                log_string('only evaluate, restored model from: \n\t%s'%MODEL_PATH)
            log_string('training is finished \n')

            if epoch%10 == 0:
                eval_log_str = eval_one_epoch(sess, ops, test_writer,epoch,eval_feed_buf_q)
                # Save the variables to disk.
                if not FLAGS.only_evaluate:
                    if (epoch > 0 and epoch % 1 == 0) or epoch == MAX_EPOCH-1:
                        save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"),global_step=epoch)
                        log_string("Model saved in file: %s" % os.path.basename(save_path))

                # if epoch == MAX_EPOCH -1:
                LOG_FOUT_FUSION.write('Epoch_id:'+str(epoch)+', Accuracy:'+str(eval_log_str)+'\n'+'\n\n' )
                LOG_FOUT_FUSION.flush()
                log_string('Accuracy is : %0.3f' % (eval_log_str))


def add_log(tot,epoch,batch_idx,loss_batch,c_TP_FN_FP,total_seen,t_batch_ls,SimpleFlag = 0):
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
    if SimpleFlag >0:
        log_str += ave_acc_str
    if  SimpleFlag >1:
        log_str += class_acc_str
    log_string(log_str)
    return log_str

def train_one_epoch(sess, ops, train_writer,epoch,train_feed_buf_q,pctx,opts):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    #log_string('----')
    num_blocks = net_provider.train_num_blocks
    if num_blocks!=None:
        num_batches = num_blocks // BATCH_SIZE
        if num_batches ==0: return ''
    else:
        num_batches = None

    total_seen = 0.0001
    loss_sum = 0.0
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
        poinr_cloud_data = []
        label_data = []
        if train_feed_buf_q == None:
            point_cloud_data, label_data, sg_bidxmaps_pl, globalb_bottom_center_xyz,fid_start_end = net_provider.get_train_batch(start_idx, end_idx)
            # point_cloud_data, label_data = data_provider._get_next_minibatch()  #cur_data,cur_label,cur_smp_weights =  net_provider.get_train_batch(start_idx,end_idx)
        else:
            if train_feed_buf_q.qsize() == 0:
                print('train_feed_buf_q.qsize == 0')
                break
            point_cloud_data, label_data = train_feed_buf_q.get()
        t1 = time.time()
        if type(point_cloud_data) == type(None):
            break # all data reading finished
        feed_dict = {ops['pointclouds_pl']: point_cloud_data,
                     ops['labels_pl']: label_data,
                     ops['globalb_bottom_center_xyz']: globalb_bottom_center_xyz,
                     ops['sg_bidxmaps_pl']: sg_bidxmaps_pl,
                     ops['is_training_pl']: is_training }

        if ISDEBUG  and  epoch == 0 and batch_idx ==5:
                pctx.trace_next_step()
                pctx.dump_next_step()
                summary, step, _, loss_val, pred_class_val, classification_loss_val, regression_loss_val, loss_details_val, accuracy_classification, recall_classification, num_positive_label \
                                       = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred_class'], ops['classification_loss'], ops['regression_loss'], ops['loss_details'], ops['accuracy_classification'], ops['recall_classification'], ops['num_positive_label']], feed_dict=feed_dict)
                pctx.profiler.profile_operations(options=opts)
        else:
            summary, step, _, loss_val, pred_class_val, classification_loss_val, regression_loss_val, loss_details_val, accuracy_classification, recall_classification, num_positive_label  \
                                       = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred_class'], ops['classification_loss'], ops['regression_loss'], ops['loss_details'], ops['accuracy_classification'], ops['recall_classification'], ops['num_positive_label']], feed_dict=feed_dict)

        t2=time.time()
        t_batch_ls.append( np.reshape(np.array([t1-t0,time.time() - t1]),(2,1)) )

        max_memory_usage = sess.run(ops['max_memory_usage'])
        # print('reading data time is:{}, the training time is {}'.format((t1-t0),(t2-t1)))

        if ISSUMMARY: train_writer.add_summary(summary, step)
        if batch_idx%80 == 0:
            # print('the training batch is {}, the loss value is {}'.format(batch_idx, loss_val))
            # print('the classificaiton loss is {}, the regression loss is {}'.format(classification_loss_val, regression_loss_val))
            # print('accuracy of classification is {}'.format(accuracy_classification ))
            # print('the details of loss value, dl:{},dw:{},dtheta:{},dx:{},dy:{}'.format(\
            #                                   loss_details_val[0], loss_details_val[1], loss_details_val[2],  loss_details_val[3], loss_details_val[4]))

            log_string('------batch is {},----------------'.format(batch_idx))
            log_string('reading data time is:{}, the training time is {}'.format((t1-t0),(t2-t1)))
            log_string('loss {},  classificaiton {}, regression {}'.format(loss_val ,classification_loss_val, regression_loss_val))
            log_string('******** AC is {}, RC is {}, PN is {}'.format(accuracy_classification, recall_classification, num_positive_label))
            log_string('########dl:{},dw:{},dtheta:{},dx:{},dy:{}'.format(\
                                               loss_details_val[0], loss_details_val[1], loss_details_val[2],  loss_details_val[3], loss_details_val[4]))
            log_string('max_memory_usage:{} \n'.format(max_memory_usage))
        if False and ( batch_idx == num_batches-1 or  (epoch == 0 and batch_idx % 20 ==0) or batch_idx%200==0) : ## not evaluation in one epoch
            pred_class_val = np.argmax(pred_class_val, 2)
            loss_sum += loss_val
            total_seen += (BATCH_SIZE*NUM_POINT)
            c_TP_FN_FP += EvaluationMetrics.get_TP_FN_FP(NUM_CLASSES,pred_class_val,cur_label)

            train_logstr = add_log('train',epoch,batch_idx,loss_sum/(batch_idx+1),c_TP_FN_FP,total_seen,t_batch_ls)
        if batch_idx == 200:
            os.system('nvidia-smi')
    return train_logstr

def limit_eval_num_batches(epoch,num_batches):
    if epoch%5 != 0:
        num_batches = min(num_batches,num_batches)
        #num_batches = min(num_batches,31)
    return num_batches

def eval_one_epoch(sess, ops, test_writer, epoch,eval_feed_buf_q):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_seen = 0.00001
    loss_sum = 0.0
    c_TP_FN_FP = np.zeros(shape=(3,NUM_CLASSES))

    log_string('----')

    num_blocks = net_provider.eval_num_blocks  # evaluation some of data
    if num_blocks != None:
        num_batches = num_blocks // BATCH_SIZE
        num_batches = limit_eval_num_batches(epoch,num_batches)
        if num_batches == 0:
            print('\ntest num_blocks=%d  BATCH_SIZE=%d  num_batches=%d'%(num_blocks,BATCH_SIZE,num_batches))
            return 0.0
    else:
        num_batches = None

    eval_logstr = ''
    t_batch_ls = []
    all_gt_box = []
    all_pred_class_val = []
    all_pred_box_val = []
    all_xyz   = []
    batch_idx = -1
    # label
    while (batch_idx < num_batches-1) or (num_batches==None):
        t0 = time.time()
        batch_idx += 1
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        if eval_feed_buf_q == None:
             point_cloud_data, gt_box, sg_bidxmaps_pl, globalb_bottom_center_xyz, fid_start_end = net_provider.get_eval_batch(start_idx, end_idx) #cur_data,cur_label,cur_smp_weights = net_provider.get_eval_batch(start_idx,end_idx)
        else:
            if eval_feed_buf_q.qsize() == 0:
                print('eval_feed_buf_q.qsize == 0')
                break
            point_cloud_data, label_data, epoch_buf = eval_feed_buf_q.get()
            #assert batch_idx == batch_idx_buf and epoch== epoch_buf
        t1 = time.time()
        if type(point_cloud_data) == type(None):
            print('batch_idx:%d, get None, reading finished'%(batch_idx))
            break # all data reading finished
        feed_dict = {ops['pointclouds_pl']: point_cloud_data,
                     ops['labels_pl']: gt_box,
                     ops['globalb_bottom_center_xyz']: globalb_bottom_center_xyz,
                     ops['sg_bidxmaps_pl']: sg_bidxmaps_pl,
                     ops['is_training_pl']: is_training }

        summary, step, loss_val, pred_class_val, pred_prob_val, pred_box_val, xyz_pl, classification_loss_val, regression_loss_val, loss_details_val, accuracy_classification, recall_classification, num_positive_label \
          = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred_class'], ops['pred_prob'], ops['pred_box'], ops['xyz_pl'], ops['classification_loss'], ops['regression_loss'], ops['loss_details'], ops['accuracy_classification'], ops['recall_classification'], ops['num_positive_label']],  feed_dict=feed_dict)

        if ISSUMMARY and  test_writer != None:
            test_writer.add_summary(summary, step)
        t_batch_ls.append( np.reshape(np.array([t1-t0,time.time() - t1]),(2,1)) )

        all_gt_box.append(gt_box)  # all_gt_box is a list, num_batches x BATCH_SIZE x ( k*8 ), all_gt_box[n][m] is the ground truth box of one label image.
        all_pred_class_val.append(pred_prob_val)  # the all_pred_class_val is the list, num_batches x BATCH_SIZE x point_num x 4, all_pred_val[n] is the narray of BATCH_SIZE x point_num
        all_pred_box_val.append(pred_box_val)  # the all_pred_box_val is also list, num_batches x BATCH_SIZE x point_num x 14, all_pred_box_val[n] is the narray of BATCH_SIZE x point_num
        all_xyz.append(xyz_pl)    # the all_xyz shape: num_batches x (BATCHSIZE X point_num x3)

        if False and (batch_idx == num_batches-1 or (FLAGS.only_evaluate and  batch_idx%30==0)):
            pred_logits = np.argmax(pred_prob_val, 2)
            total_seen += (BATCH_SIZE*NUM_POINT)
            loss_sum += loss_val
            c_TP_FN_FP += EvaluationMetrics.get_TP_FN_FP(NUM_CLASSES,pred_logits,cur_label)
            eval_logstr = add_log('eval',epoch,batch_idx,loss_sum/(batch_idx+1),c_TP_FN_FP,total_seen,t_batch_ls)
        if batch_idx%40 == 0:
            # print('the test batch is {}, the loss value is {}'.format(batch_idx, loss_val))
            # print('the classificaiton loss is {}, the regression loss is {}'.format(classification_loss_val, regression_loss_val))
            # print('the details of loss value, dl:{},dw:{},dtheta:{},dx:{},dy:{}'.format(\
            #                                   loss_details_val[0], loss_details_val[1], loss_details_val[2],  loss_details_val[3], loss_details_val[4]))
            print('------batch is {},----------------'.format(batch_idx))
            print('loss {},  classificaiton {}, regression {}'.format(loss_val ,classification_loss_val, regression_loss_val))
            print('******** AC is {}, RC is {}, PN is {}'.format(accuracy_classification, recall_classification, num_positive_label))
            print('########dl:{},dw:{},dtheta:{},dx:{},dy:{}, \n'.format(\
                                               loss_details_val[0], loss_details_val[1], loss_details_val[2],  loss_details_val[3], loss_details_val[4]))
            # print('max_memory_usage:{}'.format(max_memory_usage))

    ## estimate the all detection results
    # format of all_pred_boxes: l, w, h, theta, x, y, z, score
    # format of gt_boxes: type, l, w, h, theta, x, y, z
    # put assemble all_pred_class_val and all_pred_box_val together accroding to
    # the format of all_pred_boxes
    # using 0.05 to select the all prediction, getting all_3D_box
    # using nms_3d to filter out the all bounding box
    print('---------------------------')
    print('Start evaluation!!')
    all_pred_boxes, all_gt_boxes =  boxes_assemble_filter(all_pred_class_val, all_pred_box_val, all_xyz, all_gt_box, 0.05)

    # caculate the average precision with the detection results
    aveg_precision = evaluation_2d(all_pred_boxes, all_gt_boxes, cfg.TEST.RPN_THRESH)
    # delete the all_gt_box, all_pred_class_val and all_pred_box_val to save
    # memory
    print('The average precision is {}'.format(aveg_precision))
    #if FLAGS.only_evaluate:
    #    obj_dump_dir = os.path.join(FLAGS.log_dir,'obj_dump')
    #    net_provider.gen_gt_pred_objs(FLAGS.visu,obj_dump_dir)
    #    net_provider.write_file_accuracies(FLAGS.log_dir)
    #    print('\nobj out path:'+obj_dump_dir)

    return aveg_precision

def boxes_assemble_filter(all_pred_class_val, all_pred_box_val, all_xyz, all_gt_box , thresh = 0.05):

    #all_pred_boxes = np.zeros([1,8])  #l, w, h, theta, x, y, z, score
    all_pred_boxes = []  # saved in list
    num_batch = len(all_pred_class_val)
    batch_size = all_pred_class_val[0].shape[0]
    gt_box_ = []
    num_anchors = cfg.TRAIN.NUM_ANCHORS
    num_class  =  cfg.TRAIN.NUM_CLASSES
    num_regression = cfg.TRAIN.NUM_REGRESSION
    # generate, (num_samples x num_point) x 8
    for i in range(num_batch):
        for j in range(batch_size):
            index = i*batch_size + j
            temp_pred_class = np.array([all_pred_class_val[i][j,:,(x*num_class+1):((x+1)*num_class)] for x in range(num_anchors)]).transpose(1, 0, 2) ##shape: 512 x num_anchors x 1
            temp_pred_class = temp_pred_class.reshape(-1, 1)  # shape: n x 1

            temp_all_box = bbox_transform_inv_2d(all_pred_box_val[i][j,:,:], all_xyz[i][j,:,:])  # key function

            temp_all_       =  np.concatenate(( temp_all_box,temp_pred_class), axis=1)
            temp_all_       = temp_all_[ np.where( temp_all_[:,5] >= thresh)[0], :]  ## temp_all_ shape: n x 6: l,w,alpha,x,y, confidence
            if temp_all_.shape[0] > 0:  ## there is no prediction box whose prediction is over thresh
                temp_all_       = nms_2d(temp_all_, cfg.TEST.NMS)
                all_pred_boxes.append(temp_all_)
            gt_box_.append(all_gt_box[i][j])

    return all_pred_boxes, gt_box_


def main():
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
        LOG_FOUT_FUSION.close()
