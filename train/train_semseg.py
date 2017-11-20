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
sys.path.append(os.path.join(ROOT_DIR,'scannet'))
from pointnet2_sem_seg import  placeholder_inputs,get_model,get_loss
import provider
import get_dataset
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
parser.add_argument('--dataset_name', default='scannet', help='dataset_name: scannet, stanford_indoor')
parser.add_argument('--channel_elementes', default='xyz_1norm', help='channel_elements: xyz_1norm,xyz_midnorm,color_1norm')
FLAGS = parser.parse_args()
FLAGS.channel_elementes = FLAGS.channel_elementes.split(',')

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

            if train_one_epoch(sess, ops, train_writer,epoch) == False:
                print('get nan loss, break training')
                break
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if (epoch % 10 == 0) or (epoch > 35 and epoch % 3 == 0):
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"),global_step=epoch)
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer,epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string('----')

    num_blocks = DATASET.num_blocks['train']
    if num_blocks!=None:
        num_batches = num_blocks // BATCH_SIZE
        #num_batches = min(500,num_batches)
    else:
        num_batches = None

    total_correct = 0
    batch_correct = 0
    total_seen = 0
    loss_sum = 0

    smpws = np.ones([BATCH_SIZE,NUM_POINT],dtype=np.float32)
    print('total batch num = ',num_batches)
    t0 = time.time()
    batch_idx = -1
    def log_train():
        train_t_perbatch = (time.time() - t0) / (batch_idx+1)
        log_string('[%d-%d] train loss: %f\taccuracy(batch-total): %f - %f\tbatch t: %fs total t:%f s' %
                   (epoch,batch_idx,loss_sum / float(batch_idx+1),
                    batch_correct / float(BATCH_SIZE*NUM_POINT),total_correct / float(total_seen),
                    train_t_perbatch,time.time()-START_TIME))

    DATASET.shuffle_idx()
    while (batch_idx < num_batches) or (num_batches==None):
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
        batch_correct = np.sum(pred_val == cur_label )
        total_correct += batch_correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val

        if np.isnan(loss_val) and np.sum(pred_val)==0:
            correct_num = np.sum(cur_label==pred_val)
            log_train()
            return False
      #      print('correct_num=%d %f'%(correct_num,1.0*correct_num/cur_label.size))
      #      print('pred_val = ',pred_val[0][0:20])
      #      print('cur_label = ',cur_label[0][0:20])
      #      import pdb; pdb.set_trace()  # XXX BREAKPOINT

        if (epoch == 0 and batch_idx <= 100) or batch_idx%100==0:
            log_train()
    log_string('\n')
    log_train()
    return True


def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    log_string('----')

    num_blocks = DATASET.num_blocks['test']
    if num_blocks != None:
        num_batches = num_blocks // BATCH_SIZE
    else:
        num_batches = None

    smpws = np.ones([BATCH_SIZE,NUM_POINT],dtype=np.float32)
    t0 = time.time()

    batch_idx = -1
    while (batch_idx < num_batches) or (num_batches==None):
        batch_idx += 1
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        cur_data,cur_label,cur_smp_weights = DATASET.test_dlw(start_idx,end_idx)
        if type(cur_data) == type(None):
            break # all data reading finished
        feed_dict = {ops['pointclouds_pl']: cur_data,
                     ops['labels_pl']: cur_label,
                     ops['is_training_pl']: is_training,
                     ops['smpws_pl']: cur_smp_weights }
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == cur_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_POINT):
                l = cur_label[i, j]
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
