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
sys.path.append(os.path.join(ROOT_DIR,'scannet'))
from pointnet2_sem_seg import  placeholder_inputs,get_model,get_loss
import provider
import get_dataset
from evaluation import EvaluationMetrics
#from block_data_prep_util import Normed_H5f,Net_Provider

parser = argparse.ArgumentParser()
parser.add_argument('--channel_elementes', default='xyz_1norm', help='channel_elements: xyz_1norm,xyz_midnorm,color_1norm')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')
parser.add_argument('--test_area', type=int, default=6, help='Which area to use for test, option: 1-6 [default: 6]')
parser.add_argument('--dataset_name', default='scannet', help='dataset_name: scannet, stanford_indoor')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
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

parser.add_argument('--auto_break',action='store_true',help='If true, auto break when error occurs')

FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

if FLAGS.only_evaluate:
    MAX_EPOCH = 1
    log_name = 'log_Test.txt'
    FLAGS.test_area = -1
else:
    MAX_EPOCH = FLAGS.max_epoch
    log_name = 'log_Train.txt'
    FLAGS.log_dir = FLAGS.log_dir+str(FLAGS.test_area)+'-B'+str(BATCH_SIZE)+'-'+\
                    FLAGS.channel_elementes+'-'+str(NUM_POINT)+'-'+FLAGS.dataset_name
FLAGS.channel_elementes = FLAGS.channel_elementes.split(',')

LOG_DIR = os.path.join(ROOT_DIR,'train_res/semseg_result/'+FLAGS.log_dir)
FLAGS.model_path = os.path.join(LOG_DIR,'model.ckpt')
MODEL_PATH = FLAGS.model_path
LOG_DIR_FUSION = os.path.join(ROOT_DIR,'train_res/semseg_result/fusion_log.txt')
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
os.system('cp ../models/pointnet2_sem_seg.py %s' % (LOG_DIR)) # bkp of model def
os.system('cp train_semseg.py %s' % (LOG_DIR)) # bkp of train procedure
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
DATASET = get_dataset.GetDataset( FLAGS.dataset_name,NUM_POINT,
                                 test_area=FLAGS.test_area,
                                 max_test_fn=FLAGS.max_test_file_num,
                                 channel_elementes=FLAGS.channel_elementes )
NUM_CLASSES = DATASET.num_classes

START_TIME = time.time()

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
log_string(DATASET.data_sum_str+'\n')

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

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            NUM_CHANNELS = DATASET.num_channels
            pointclouds_pl, labels_pl,smpws_pl = placeholder_inputs(BATCH_SIZE, NUM_POINT,NUM_CHANNELS)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred,end_points = get_model(pointclouds_pl, is_training_pl, NUM_CLASSES, bn_decay=bn_decay)
            loss = get_loss(pred, labels_pl,smpws_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
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
            saver = tf.train.Saver()

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

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'smpws_pl': smpws_pl}
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            if not FLAGS.only_evaluate:
                train_log_str = train_one_epoch(sess, ops, train_writer,epoch)
            else:
                train_log_str = ''
                saver.restore(sess,MODEL_PATH)
                log_string('restored model from: \n\t%s'%MODEL_PATH)
            eval_log_str = eval_one_epoch(sess, ops, test_writer,epoch)

            # Save the variables to disk.
            if not FLAGS.only_evaluate:
                if (epoch > 0 and epoch % 10 == 0) or (epoch > 35 and epoch % 3 == 0) or epoch == MAX_EPOCH-1:
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"),global_step=epoch)
                    log_string("Model saved in file: %s" % os.path.basename(save_path))

            if epoch == MAX_EPOCH -1:
                LOG_FOUT_FUSION.write( str(FLAGS)+'\n\n'+train_log_str+'\n'+eval_log_str+'\n\n' )


def add_log(tot,epoch,batch_idx,loss_sum,c_TP_FN_FP,total_seen,t_batch_ls,SimpleFlag = 0):
    ave_whole_acc,class_acc_str,ave_acc_str = EvaluationMetrics.get_class_accuracy(
                                c_TP_FN_FP,total_seen)
    log_str = ''
    if len(t_batch_ls)>0:
        t_per_batch = np.mean(np.array(t_batch_ls))
        t_per_block = t_per_batch / BATCH_SIZE
        #t_per_point = t_per_block / NUM_POINT * 1000
    else:
        t_per_block = -1
    log_str += '%s [%d - %d] \t t_block:%0.3f\tloss: %0.3f \tacc: %0.3f' % \
            ( tot,epoch,batch_idx,t_per_block,loss_sum / float(batch_idx+1),ave_whole_acc )
    if SimpleFlag >0:
        log_str += ave_acc_str
    if  SimpleFlag >1:
        log_str += class_acc_str
    log_string(log_str)
    return log_str

def train_one_epoch(sess, ops, train_writer,epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    #log_string('----')
    num_blocks = DATASET.num_blocks['train']
    if num_blocks!=None:
        num_batches = num_blocks // BATCH_SIZE
        if num_batches ==0: return ''
    else:
        num_batches = None

    total_seen = 0.0001
    loss_sum = 0
    c_TP_FN_FP = np.zeros(shape=(3,NUM_CLASSES))

    print('total batch num = ',num_batches)
    batch_idx = -1

    DATASET.shuffle_idx()
    t_batch_ls=[]
    while (batch_idx < num_batches) or (num_batches==None):
        t0 = time.time()
        batch_idx += 1
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        cur_data,cur_label,cur_smp_weights = DATASET.train_dlw(start_idx,end_idx)
        if type(cur_data) == type(None):
            break # all data reading finished
        feed_dict = {ops['pointclouds_pl']: cur_data,
                     ops['labels_pl']: cur_label,
                     ops['is_training_pl']: is_training,
                     ops['smpws_pl']: cur_smp_weights}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val

        c_TP_FN_FP += EvaluationMetrics.get_TP_FN_FP(NUM_CLASSES,pred_val,cur_label)

        t_batch_ls.append( time.time() - t0 )
        if (epoch == 0 and batch_idx <= 100) or batch_idx%100==0:
            add_log('train',epoch,batch_idx,loss_sum,c_TP_FN_FP,total_seen,t_batch_ls)
    return add_log('train',epoch,batch_idx,loss_sum,c_TP_FN_FP,total_seen,t_batch_ls)

def eval_one_epoch(sess, ops, test_writer, epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_seen = 0.00001
    loss_sum = 0
    c_TP_FN_FP = np.zeros(shape=(3,NUM_CLASSES))

    log_string('----')

    num_blocks = DATASET.num_blocks['test']
    if num_blocks != None:
        num_batches = num_blocks // BATCH_SIZE
        if num_batches == 0:
            print('\ntest num_blocks=%d  BATCH_SIZE=%d  num_batches=%d'%(num_blocks,BATCH_SIZE,num_batches))
            return ''
    else:
        num_batches = None

    t_batch_ls = []
    batch_idx = -1
    while (batch_idx < num_batches) or (num_batches==None):
        t0 = time.time()
        batch_idx += 1
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        cur_data,cur_label,cur_smp_weights = DATASET.test_dlw(start_idx,end_idx)
        if type(cur_data) == type(None):
            print('batch_idx:%d, get None, reading finished'%(batch_idx))
            break # all data reading finished
        feed_dict = {ops['pointclouds_pl']: cur_data,
                     ops['labels_pl']: cur_label,
                     ops['is_training_pl']: is_training,
                     ops['smpws_pl']: cur_smp_weights }
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        if test_writer != None:
            test_writer.add_summary(summary, step)
        pred_logits = np.argmax(pred_val, 2)
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val
        c_TP_FN_FP += EvaluationMetrics.get_TP_FN_FP(NUM_CLASSES,pred_logits,cur_label)
        t_batch_ls.append( time.time() - t0 )
        if FLAGS.only_evaluate:
            #DATASET.write_pred(pred_val)
            if batch_idx%10==0:
                add_log('eval',epoch,batch_idx,loss_sum,c_TP_FN_FP,total_seen,t_batch_ls)


    return add_log('eval',epoch,batch_idx,loss_sum,c_TP_FN_FP,total_seen,t_batch_ls)

if __name__ == "__main__":
    if FLAGS.auto_break:
        try:
            train()
            LOG_FOUT.close()
        except:
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
    else:
        train()
        LOG_FOUT.close()
