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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR,'utils'))
sys.path.append(os.path.join(ROOT_DIR,'utils_xyz'))
sys.path.append(os.path.join(ROOT_DIR,'models'))
sys.path.append(os.path.join(ROOT_DIR,'config'))
from pointnet2_obj_detection_tf4 import  placeholder_inputs,get_model,get_loss
#import provider
import get_dataset
from evaluation import EvaluationMetrics
from kitti_data_net_provider import kitti_data_net_provider #Normed_H5f,Net_Provider
from config import cfg
import multiprocessing as mp
from bbox_transform import bbox_transform_inv
from nms_3d import nms_3d
from evaluation_3d import evaluation_3d
from ply_util import create_ply, gen_box_pl



ISDEBUG = False
ISSUMMARY = False

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', default='rawh5_kitti', help='rawh5_kitti')
#parser.add_argument('--all_fn_globs', type=str,default='stride_1_step_2_8192_normed/',\
#                    help='The file name glob for both training and evaluation')
parser.add_argument('--feed_elements', default='xyz_raw', help='xyz_1norm,xyz_midnorm,color_1norm')
parser.add_argument('--batch_size', type=int, default= 32, help='Batch Size during training [default: 24]')
parser.add_argument('--eval_fnglob_or_rate',  default='train', help='file name str glob or file number rate: scan1*.nh5 0.2')
parser.add_argument('--num_point', type=int, default=2**15, help='Point number [default: 2**15]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')

parser.add_argument('--num_batches', type=int, default= 2, help='decides how many visulization data you want to generate')


parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate [default: 0.01]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--max_test_file_num', type=int, default=None, help='Which area to use for test, option: 1-6 [default: 6]')

parser.add_argument('--only_evaluate',action='store_true',help='do not train')
parser.add_argument('--finetune',action='store_true',help='do not train')
parser.add_argument('--model_epoch', type=int, default=10, help='the epoch of model to be restored')

parser.add_argument('--auto_break',action='store_true',help='If true, auto break when error occurs')

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

NUM_BATCH = FLAGS.num_batches

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
    FLAGS.log_dir = FLAGS.log_dir+'-B'+str(BATCH_SIZE)+'-'+\
                    FLAGS.feed_elements+'-'+str(NUM_POINT)+'-'+FLAGS.dataset_name+'-eval_'+log_eval_fn_glob
FLAGS.feed_elements = FLAGS.feed_elements.split(',')

LOG_DIR = os.path.join(ROOT_DIR,'train_res/object_detection_result/'+FLAGS.log_dir)
MODEL_PATH = os.path.join(LOG_DIR,'model.ckpt-'+str(FLAGS.model_epoch))
LOG_DIR_FUSION = os.path.join(ROOT_DIR,'train_res/object_detection_result/fusion_log.txt')
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
os.system('cp %s/models/pointnet2_obj_detection.py %s' % (ROOT_DIR,LOG_DIR)) # bkp of model def
os.system('cp %s/train_obj_detection.py %s' % (BASE_DIR,LOG_DIR)) # bkp of train procedure
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
# FLAGS.all_fn_globs = FLAGS.all_fn_globs.split(',')
#net_provider = Net_Provider(dataset_name=FLAGS.dataset_name, \
#                            all_filename_glob=FLAGS.all_fn_globs, \
#                            eval_fnglob_or_rate=FLAGS.eval_fnglob_or_rate,\
#                            only_evaluate = FLAGS.only_evaluate,\
#                            num_point_block = NUM_POINT,
#                            feed_elements=FLAGS.feed_elements)
data_provider = kitti_data_net_provider(DATASET_NAME,BATCH_SIZE)
NUM_CHANNELS = cfg.TRAIN.NUM_CHANNELS  # x, y, z
NUM_CLASSES =  cfg.TRAIN.NUM_CLASSES   # bg, fg
NUM_REGRESSION = cfg.TRAIN.NUM_REGRESSION

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
            pointclouds_pl, labels_pl, smpws_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT,NUM_CHANNELS, NUM_REGRESSION)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            end_points, pred_class, pred_box, xyz_pl = get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
            loss, classification_loss, regression_loss, pred_prob = get_loss(BATCH_SIZE,pred_class, pred_box, labels_pl,smpws_pl, xyz_pl)
            tf.summary.scalar('loss', loss)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
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

        # define operations
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred_class': pred_class,
               'pred_box': pred_box,
               'xyz_pl': xyz_pl,
               'loss': loss,
               'classification_loss':classification_loss,
               'regression_loss':regression_loss,
               'pred_prob':pred_prob,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'smpws_pl': smpws_pl}
        if FLAGS.finetune:
            saver.restore(sess,MODEL_PATH)
            log_string('finetune, restored model from: \n\t%s'%MODEL_PATH)

        log_string(data_provider.data_summary_str)

        epoch_start = 0
        if FLAGS.finetune:
            epoch_start+=(FLAGS.model_epoch+1)
        for epoch in range(epoch_start,epoch_start+MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            #if train_feed_buf_q == None:
            #    net_provider.update_train_eval_shuffled_idx()
            train_log_str = ''
            saver.restore(sess,MODEL_PATH)
            log_string('only evaluate, restored model from: \n\t%s'%MODEL_PATH)
            eval_log_str = eval_one_epoch(sess, ops, test_writer,epoch,eval_feed_buf_q)


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
    num_blocks = data_provider.num_train_data
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
        #start_idx = batch_idx * BATCH_SIZE
        #end_idx = (batch_idx+1) * BATCH_SIZE
        poinr_cloud_data = []
        label_data = []
        if train_feed_buf_q == None:
            point_cloud_data, label_data = data_provider._get_next_minibatch()  #cur_data,cur_label,cur_smp_weights =  net_provider.get_train_batch(start_idx,end_idx)
        else:
            if train_feed_buf_q.qsize() == 0:
                print('train_feed_buf_q.qsize == 0')
                break
            #cur_data,cur_label,cur_smp_weights, batch_idx_buf,epoch_buf = train_feed_buf_q.get()
            point_cloud_data, label_data = train_feed_buf_q.get()
        cur_smp_weights = np.ones((point_cloud_data.shape[0], point_cloud_data.shape[1]))
        t1 = time.time()
        if type(point_cloud_data) == type(None):
            break # all data reading finished
        feed_dict = {ops['pointclouds_pl']: point_cloud_data,
                     ops['labels_pl']: label_data,
                     ops['is_training_pl']: is_training,
                     ops['smpws_pl']: cur_smp_weights}

        if ISDEBUG  and  epoch == 0 and batch_idx ==5:
                pctx.trace_next_step()
                pctx.dump_next_step()
                summary, step, _, loss_val, pred_class_val, classification_loss_val, regression_loss_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred_class'], ops['classification_loss'], ops['regression_loss']],
                                            feed_dict=feed_dict)
                pctx.profiler.profile_operations(options=opts)
        else:
            summary, step, _, loss_val, pred_class_val, classification_loss_val, regression_loss_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred_class'], ops['classification_loss'], ops['regression_loss']],
                                        feed_dict=feed_dict)

        t_batch_ls.append( np.reshape(np.array([t1-t0,time.time() - t1]),(2,1)) )
        if ISSUMMARY: train_writer.add_summary(summary, step)
        if batch_idx%80 == 0:
            print('the training batch is {}, the loss value is {}'.format(batch_idx, loss_val))
            print('the classificaiton loss is {}, the regression loss is {}'.format(classification_loss_val, regression_loss_val))
            #print('the all merged is {}'.format(summary))
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
        num_batches = min(num_batches,31)
    return num_batches

def eval_one_epoch(sess, ops, test_writer, epoch,eval_feed_buf_q):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_seen = 0.00001

    log_string('----')

    num_batches = NUM_BATCH

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
            point_cloud_data, label_data, gt_box = data_provider._get_evaluation_minibatch(start_idx, end_idx) #cur_data,cur_label,cur_smp_weights = net_provider.get_eval_batch(start_idx,end_idx)
        else:
            if eval_feed_buf_q.qsize() == 0:
                print('eval_feed_buf_q.qsize == 0')
                break
            point_cloud_data, label_data, epoch_buf = eval_feed_buf_q.get()
            #assert batch_idx == batch_idx_buf and epoch== epoch_buf
        cur_smp_weights = np.ones((point_cloud_data.shape[0], point_cloud_data.shape[1]))
        t1 = time.time()
        print('time of reading is {}'.format(t1-t0))
        if type(point_cloud_data) == type(None):
            print('batch_idx:%d, get None, reading finished'%(batch_idx))
            break # all data reading finished
        feed_dict = {ops['pointclouds_pl']: point_cloud_data,
                     ops['labels_pl']: label_data,
                     ops['is_training_pl']: is_training,
                     ops['smpws_pl']: cur_smp_weights }
        summary, step, loss_val, pred_class_val, pred_prob_val, pred_box_val, xyz_pl, classification_loss_val, regression_loss_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred_class'], ops['pred_prob'], ops['pred_box'], ops['xyz_pl'], ops['classification_loss'], ops['regression_loss']],
                                      feed_dict=feed_dict)

        ## generating the raw point cloud and downsampled point cloud
        color_1 = np.array([[0, 0, 0]])
        color_1 = np.tile(color_1, (point_cloud_data.shape[1],1))
        raw_point_cloud_color = np.concatenate((point_cloud_data[0], color_1),1)
        color_2 = np.array([[255, 255, 255]])
        color_2 = np.tile(color_2, (xyz_pl.shape[1], 1))
        xyz_pl[0] = xyz_pl[0] + np.array([[0.05, 0.05, 0.05]])
        xyz_pl_color = np.concatenate((xyz_pl[0], color_2),1)

        xyz_color = np.concatenate((raw_point_cloud_color, xyz_pl_color),0)
        path_vis = os.path.join(ROOT_DIR,'data/visulization/','raw_xyz_'+str(batch_idx)+'.ply')
        create_ply(xyz_color, path_vis)

        t2 = time.time()
        print('time of generating is {}'.format(t2 - t1))
        #create_ply(xyz_pl_color, path_vis)
        ## generating the raw point cloud and ground truth bounding boxes

        gt_box_ = get_box_coordinate(gt_box[0][:,1:8])
        path_vis = os.path.join(ROOT_DIR,'data/visulization/','raw_gt_box_'+str(batch_idx)+'.ply')
        gen_box_pl(path_vis, gt_box_, point_cloud_data[0])

        t3 = time.time()
        print('time of ground truth box is {}'.format(t3 - t2))
        ## generating the raw point cloud and predicted bounding boxes

        num_anchors = cfg.TRAIN.NUM_ANCHORS
        num_class  =  cfg.TRAIN.NUM_CLASSES
        pred_box_ = bbox_transform_inv(pred_box_val[0], xyz_pl[0])
        pred_class = np.array([pred_class_val[0, :,(x*num_class+1):((x+1)*num_class)] for x in range(num_anchors)]).transpose(1, 0, 2) ##shape: 512 x num_anchors x 1
        pred_class = pred_class.reshape(-1, 1)
        pred_box_ =  np.concatenate(( pred_box_, pred_class), axis=1)
        pred_box_ = pred_box_[ np.where( pred_box_[:,7] >= 0.2)[0], :]
        if pred_box_.shape[0]>0:
            pred_box_ = nms_3d( pred_box_, cfg.TEST.NMS)

        pred_box_ = get_box_coordinate(pred_box_[:,0:7])

        path_vis = os.path.join(ROOT_DIR,'data/visulization/','raw_pred_box_'+str(batch_idx)+'.ply')
        gen_box_pl(path_vis, pred_box_, point_cloud_data[0])

        t4 = time.time()
        print('time of predicting box is {}'.format(t4 - t3))

        if batch_idx%40 == 0:
            print('the test batch is {}, the loss value is {}'.format(batch_idx, loss_val))
            print('the classificaiton loss is {}, the regression loss is {}'.format(classification_loss_val, regression_loss_val))

    print('Done!!')
    return 1


def get_box_coordinate(all_boxes):
    ## transform the bounding box into array of coordinates
    assert all_boxes.shape[0]>0
    assert all_boxes.shape[1]==7
    num_box = all_boxes.shape[0]
    all_coordinates = np.zeros((1,3))
    for i in range(num_box):
        l = all_boxes[i, 0]
        w = all_boxes[i, 1]
        h = all_boxes[i, 2]
        xy = np.array([[-w/2.0, l/2.0],[w/2.0, l/2.0],[w/2.0, -l/2.0],[-w/2.0, -l/2.0]])
        theta = - all_boxes[i, 3]
        T  = np.array([[np.cos(theta), np.sin(theta) ],[-np.sin(theta), np.cos(theta)]])
        a_xy = np.dot(xy, T)
        xyz_0 = np.concatenate((a_xy, np.zeros((xy.shape[0],1))), 1)
        xyz_1 = np.concatenate((a_xy, np.full((xy.shape[0],1), h)), 1)
        xyz_all = np.concatenate((xyz_0,xyz_1), 0)
        xyz_all = xyz_all + np.array([[all_boxes[i,4], all_boxes[i, 5], all_boxes[i, 6]]])
        all_coordinates = np.append(all_coordinates, xyz_all, 0)

    return all_coordinates[1:,:]

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
            '''
            # l, w, h, alpha, x, y ,z
            temp_pred_box_l   = np.array([ np.exp(all_pred_box_val[i][j,:,(x*num_regression)])*anchor_length  for x in range(num_anchors)])
            temp_pred_box_l   = temp_pred_box_l.reshape(-1,1)
            temp_pred_box_w   = np.array([ np.  exp(all_pred_box_val[i][j,:,(x*num_regression+1)])*anchor_width  for x in range(num_anchors)])
            temp_pred_box_w   = temp_pred_box_w.reshape(-1,1)
            temp_pred_box_h   = np.array([ np.exp(all_pred_box_val[i][j,:,(x*num_regression+2)])*anchor_height  for x in range(num_anchors)])
            temp_pred_box_h   = temp_pred_box_h.reshape(-1,1)
            temp_pred_box_alpha = np.array([ all_pred_box_val[i][j,:,(x*num_regression+3)]*np.pi/4+anchor_alpha[x,0]  for x in range(num_anchors)])
            temp_pred_box_alpha = temp_pred_box_alpha.reshape(-1,1)
            temp_pred_box_x   = np.array([ all_pred_box_val[i][j,:,(x*num_regression+4)]*anchor_length + all_xyz[i][j,:,0]  for x in range(num_anchors) ])
            temp_pred_box_x   = temp_pred_box_x.reshape(-1,1)
            temp_pred_box_y   = np.array([ all_pred_box_val[i][j,:,(x*num_regression+5)]*anchor_width + all_xyz[i][j,:,1]  for x in range(num_anchors) ])
            temp_pred_box_y   = temp_pred_box_y.reshape(-1,1)
            temp_pred_box_z   = np.array([ all_pred_box_val[i][j,:,(x*num_regression+6)]*anchor_height + all_xyz[i][j,:,3]  for x in range(num_anchors) ])
            temp_pred_box_z   = temp_pred_box_z.reshape(-1,1)
            '''
            # temp_pred_box   = np.array([all_pred_box_val[i][j,:,(x*num_regression):((x+1)*num_regression)] for x in range(num_anchors)]).transpose(1,0,2) ## shape: 512 x num_anchors x 7
            # temp_pred_box   = temp_pred_box.reshape(-1, num_regression) # shape: n x 7
            ## transform the prediction into real num
            temp_all_box = bbox_transform_inv(all_pred_box_val[i][j,:,:], all_xyz[i][j,:,:])

            #temp_index      = np.full((temp_pred_class.shape[0],1), index) # shape: n x 1
            # temp_all_       = np.concatenate((temp_index, temp_pred_box_l, temp_pred_box_w, temp_pred_box_h, temp_pred_box_alpha, temp_pred_box_x, temp_pred_box_y, temp_pred_box_z,  temp_pred_class),axis=1) # shape: n x 9
            temp_all_       =  np.concatenate(( temp_all_box,temp_pred_class), axis=1)
            ## getting box whose confidence is over thresh
            temp_all_       = temp_all_[ np.where( temp_all_[:,7] >= thresh)[0], :]  ## temp_all_ shape: n x 8
            ## useing nms
            if temp_all_.shape[0] > 0:  ## there is no prediction box whose prediction is over thresh
                temp_all_       = nms_3d(temp_all_, cfg.TEST.NMS)
                all_pred_boxes.append(temp_all_)
            gt_box_.append(all_gt_box[i][j])
    # all_pred_boxes = np.delete(all_pred_boxes, 0, 0)
    # all_pred_boxes = all_pred_boxes[ np.where( all_pred_boxes[:,8] >= thresh)[0], :]

    return all_pred_boxes, gt_box_

def add_train_feed_buf(train_feed_buf_q):
    with tf.device('/cpu:0'):
        max_buf_size = 20
        num_blocks = data_provider.num_train_data  #num_blocks = net_provider.train_num_blocks
        if num_blocks!=None:
            num_batches = num_blocks // BATCH_SIZE
        else:
            num_batches = None

        epoch_start = 0
        if FLAGS.finetune:
            epoch_start+=(FLAGS.model_epoch+1)
        for epoch in range(epoch_start,epoch_start+MAX_EPOCH):
            # net_provider.update_train_eval_shuffled_idx()
            batch_idx = -1
            while (batch_idx < num_batches-1) or (num_batches==None):
                if train_feed_buf_q.qsize() < max_buf_size:
                    batch_idx += 1
                    start_idx = batch_idx * BATCH_SIZE
                    end_idx = (batch_idx+1) * BATCH_SIZE
                    point_cloud_data, label_data = data_provider._get_next_minibatch()   #cur_data,cur_label,cur_smp_weights = net_provider.get_train_batch(start_idx,end_idx)
                    train_feed_buf_q.put( [cur_data,cur_label,cur_smp_weights, batch_idx,epoch] )
                    if type(cur_data) == type(None):
                        print('add_train_feed_buf: get None data from net_provider, all data put finished. epoch= %d, batch_idx= %d'%(epoch,batch_idx))
                        break # all data reading finished
                else:
                    time.sleep(0.1*BATCH_SIZE*max_buf_size/3)
            print('add_train_feed_buf: data reading finished. epoch= %d, batch_idx= %d'%(epoch,batch_idx))

def add_eval_feed_buf(eval_feed_buf_q):
    with tf.device('/cpu:1'):
        max_buf_size = 20
        num_blocks = data_provider.evaluation_num
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
                    point_cloud_data, label_data = data_provider._get_evaluation_minibatch(start_idx, end_idx)  #cur_data,cur_label,cur_smp_weights = net_provider.get_eval_batch(start_idx,end_idx)
                    eval_feed_buf_q.put( [cur_data,cur_label,cur_smp_weights, batch_idx,epoch] )
                    if type(cur_data) == type(None):
                        print('add_eval_feed_buf: get None data from net_provider, all data put finished. epoch= %d, batch_idx= %d'%(epoch,batch_idx))
                        break # all data reading finished
                else:
                    time.sleep(0.1*BATCH_SIZE*max_buf_size/3)
            print('add_eval_feed_buf: data reading finished. epoch= %d, batch_idx= %d'%(epoch,batch_idx))


def main():

    IsFeedData_MultiProcessing = False and (not FLAGS.auto_break)

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
