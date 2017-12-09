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
sys.path.append(os.path.join(ROOT_DIR,'models'))
from pointnet2_sem_seg import  placeholder_inputs,get_model,get_loss
import provider
#from block_data_prep_util import Normed_H5f,Net_Provider

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=24, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--test_area', type=int, default=6, help='Which area to use for test, option: 1-6 [default: 6]')
parser.add_argument('--max_test_file_num', type=int, default=None, help='Which area to use for test, option: 1-6 [default: 6]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
NUM_POINT = FLAGS.num_point
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

LOG_DIR = os.path.join(ROOT_DIR,'train_res/semseg_result/'+FLAGS.log_dir)
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
#os.system('cp model.py %s' % (LOG_DIR)) # bkp of model def
#os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 13

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()


def get_channel_indexs(channel_elementes):
    channel_indexs_dict = {
        'xyz_midnorm':[0,1,2],
        'color_1norm':[3,4,5],
        'xyz_1norm':  [6,7,8]}
    channel_indexs = []
    for e in channel_elementes:
        channel_indexs += channel_indexs_dict[e]
    channel_indexs.sort()
    num_channels = len(channel_indexs)
    return channel_indexs,num_channels

channel_elementes = ['xyz_1norm']
CHANNEL_INDEXS,NUM_CHANNELS = get_channel_indexs(channel_elementes)

ALL_FILES = provider.getDataFiles('../data/indoor3d_sem_seg_hdf5_data/all_files.txt')
room_filelist = [line.rstrip() for line in open('../data/indoor3d_sem_seg_hdf5_data/room_filelist.txt')]

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
# Load ALL data
data_batch_list = []
label_batch_list = []
for i,h5_filename in enumerate(ALL_FILES):
    data_batch, label_batch = provider.loadDataFile(h5_filename)
    data_batch = data_batch[...,CHANNEL_INDEXS]
    data_batch_list.append(data_batch)
    label_batch_list.append(label_batch)

data_batches = np.concatenate(data_batch_list, 0)
label_batches = np.concatenate(label_batch_list, 0)

test_area = 'Area_'+str(FLAGS.test_area)

train_idxs = []
test_idxs = []
for i,room_name in enumerate(room_filelist):
    if test_area in room_name:
        test_idxs.append(i)
    else:
        train_idxs.append(i)

if FLAGS.max_test_file_num != None and len(test_idxs)>FLAGS.max_test_file_num:
    max_test = FLAGS.max_test_file_num
    if len(test_idxs)>max_test:
        test_idxs = test_idxs[0:max_test]
    if len(train_idxs)>max_test*5:
        train_idxs = train_idxs[0:max_test*5]
    log_string('\n!!! in  small data scale: only read %d test files\n'%(FLAGS.max_test_file_num))

train_data = data_batches[train_idxs,...]
train_label = label_batches[train_idxs]
test_data = data_batches[test_idxs,...]
test_label = label_batches[test_idxs]

START_TIME = time.time()

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
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

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

            train_one_epoch(sess, ops, train_writer,epoch)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if (epoch % 10 == 0) or (epoch > 35 and epoch % 3 == 0):
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"),global_step=epoch)
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer,epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string('----')
    #current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_POINT,:], train_label[:,0:NUM_POINT])
    current_data = train_data[:,0:NUM_POINT,:]
    current_label = train_label[:,0:NUM_POINT]

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE
    num_batches = min(200,num_batches)

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    smpws = np.ones([BATCH_SIZE,NUM_POINT],dtype=np.float32)
    print('total batch num = ',num_batches)
    t0 = time.time()
    batch_idx = 0
    def log_train():
        train_t_perbatch = (time.time() - t0) / (batch_idx+1)
        log_string('[%d/%d] train loss: %f\taccuracy: %f\tbatch t: %fs total t:%f s' %
                   (epoch,batch_idx,loss_sum / float(num_batches),
                    total_correct / float(total_seen),
                    train_t_perbatch,time.time()-START_TIME))

    for batch_idx in range(num_batches):
        #if batch_idx % 100 == 0:
            #print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,
                     ops['smpws_pl']: smpws}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val

        if (epoch == 0 and batch_idx <= 100) or batch_idx%100==0:
            log_train()
    log_string('\n')
    log_train()


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    log_string('----')
    current_data = test_data[:,0:NUM_POINT,:]
    current_label = np.squeeze(test_label[:,0:NUM_POINT])

    file_size = current_data.shape[0]
    num_batches = file_size // BATCH_SIZE

    smpws = np.ones([BATCH_SIZE,NUM_POINT],dtype=np.float32)
    t0 = time.time()
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,
                     ops['smpws_pl']: smpws}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx, j] == l)
    eval_t = time.time() - t0


    log_string('batch %d eval  loss: %f\taccuracy: %f\tbatch t: %fs point t:%f ms' %
                    (batch_idx,
                    loss_sum / float(total_seen/NUM_POINT),
                    total_correct / float(total_seen),
                    eval_t/num_batches,eval_t/num_batches/NUM_POINT*1000))
    class_accuracies = np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)
    log_string('eval class accuracies: %s' % (np.array_str(class_accuracies)))
    log_string('eval avg class acc: %f' % (np.mean(class_accuracies)))



if __name__ == "__main__":
    train()
    LOG_FOUT.close()
