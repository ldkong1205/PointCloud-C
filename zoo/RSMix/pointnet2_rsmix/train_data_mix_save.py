'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import modelnet_dataset
# from modelnet_h5_dataset_data_mix_save import * 
import modelnet_h5_dataset_data_mix_save as modelnet_h5_dataset
# import modelnet_h5_dataset
import time

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name [default: pointnet2_cls_ssg]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')

# add argument
parser.add_argument('--seed', type=int, default=1, help='seed for experiment [default: 1]')
parser.add_argument('--rsmix_prob', type=float, default=0.5, help='point mix probability')
parser.add_argument('--beta', type=float, default=0.0, help='scalar value for beta function')
parser.add_argument('--convda', action='store_true', help='conventional data augmentation')
parser.add_argument('--rddrop', action='store_true', help='random point drop data augmentation')
parser.add_argument('--n_sample', type=int, default=512, help='max sample for point mix [default: 512]')
parser.add_argument('--shuffle', action='store_true', help='shuffle data augmentation')
parser.add_argument('--jitter', action='store_true', help='jitter data augmentation')
parser.add_argument('--rot', action='store_true', help='rot data augmentation')
parser.add_argument('--rdscale', action='store_true', help='rdscale data augmentation')
parser.add_argument('--shift', action='store_true', help='shift data augmentation')
parser.add_argument('--modelnet10', action='store_true', help='use modelnet10')
parser.add_argument('--mixed_data_dir', default='./data_mixed', help='mixed data dir [default: ./data_mixed]')
parser.add_argument('--mixed_data_save', action='store_true', help='mix_data_save')


FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')


BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

# NUM_CLASSES = 40

CONVDA=FLAGS.convda
RDDROP=FLAGS.rddrop
RSMIX_PROB=FLAGS.rsmix_prob
BETA=FLAGS.beta
N_SAMPLE=FLAGS.n_sample

SHUFFLE=FLAGS.shuffle
JITTER=FLAGS.jitter
ROT=FLAGS.rot
RDSCALE=FLAGS.rdscale
SHIFT=FLAGS.shift

MODELNET10=FLAGS.modelnet10
if MODELNET10:
    NUM_CLASSES=10
else:
    NUM_CLASSES=40
    

SEED = FLAGS.seed
provider.set_random_seed(SEED)


MIXED_DATA_DIR=FLAGS.mixed_data_dir
MIXED_DATA_SAVE=FLAGS.mixed_data_save

EXP_NAME = LOG_DIR.split('/')[2]
if MIXED_DATA_SAVE:
    MIXED_SAVE_DIR = os.path.join(MIXED_DATA_DIR,EXP_NAME)
else:
    MIXED_SAVE_DIR = './data_mixed'
    

if not os.path.exists(MIXED_SAVE_DIR):
    if not os.path.exists(MIXED_DATA_DIR):
        os.mkdir(MIXED_DATA_DIR)
    os.mkdir(MIXED_SAVE_DIR)


# Shapenet official train/test split
if FLAGS.normal or MODELNET10:
    assert(NUM_POINT<=10000)
    DATA_PATH = os.path.join(ROOT_DIR, 'data/modelnet40_normal_resampled')
    TRAIN_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='train', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE, modelnet10=MODELNET10)
    TEST_DATASET = modelnet_dataset.ModelNetDataset(root=DATA_PATH, npoints=NUM_POINT, split='test', normal_channel=FLAGS.normal, batch_size=BATCH_SIZE, modelnet10=MODELNET10)
else:
    assert(NUM_POINT<=2048)
    TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=True)
    TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)

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
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
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
            pointclouds_pl, labels_pl, labels_pl_b, lam = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay, class_num=NUM_CLASSES)
            # _, loss_a, loss_b, loss_a_lam, loss_b_lam = MODEL.get_loss(pred, labels_pl, end_points, labels_pl_b, lam)
            MODEL.get_loss(pred, labels_pl, end_points, labels_pl_b, lam)
            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)
            for l in losses + [total_loss]:
                tf.summary.scalar(l.op.name, l)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)
            
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
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points,
               'labels_pl_b': labels_pl_b,
               'lam': lam}

        best_acc = -1
        best_clss_acc = -1
        conv_epoch = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            # train_one_epoch(sess, ops, train_writer, loss_a, loss_b, loss_a_lam, loss_b_lam)
            train_one_epoch(sess, ops, train_writer, epoch, MIXED_SAVE_DIR, MIXED_DATA_SAVE)
            eval_accuracy, eval_class_accuracy = eval_one_epoch(sess, ops, test_writer)
            
            # # Save the variables to disk.
            # if epoch % 10 == 0:
            #     save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
            #     log_string("Model saved in file: %s" % save_path)
            
            # Save the best model
            if best_acc < eval_accuracy:
                best_acc = eval_accuracy
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)
                conv_epoch = epoch
            # if best_clss_acc < eval_class_accuracy:
                best_class_acc = eval_class_accuracy
                # save_path_class = saver.save(sess, os.path.join(LOG_DIR, "model_class_acc.ckpt"))
                # log_string("Model class_acc saved in file: %s" % save_path_class)
            log_string('>>> best accuracy : %f' %(best_acc))
            log_string('>>> at that time, best class accuracy : %f' %(best_class_acc))
        
        # measure the execution time
        execution_time = time.time()-start_time
        hour = execution_time//3600
        minute = (execution_time-hour*3600)//60
        second = execution_time-hour*3600-minute*60
        log_string('... End of the Training ...')
        log_string("trainig time : %.2f sec, %d min, %d hour" %(float(second), int(minute), int(hour)))
        log_string('*** best accuracy *** - %f' %(best_acc))
        log_string('*** at that time, best class accuracy *** -  %f' %(best_class_acc))
        log_string('*** conv epoch *** - %d' %(conv_epoch))



# def train_one_epoch(sess, ops, train_writer, loss_a, loss_b, loss_a_lam, loss_b_lam):
def train_one_epoch(sess, ops, train_writer, epoch, MIXED_SAVE_DIR, MIXED_DATA_SAVE):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TRAIN_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)
    cur_batch_label_b = np.zeros((BATCH_SIZE), dtype=np.int32)
    cur_lam = np.zeros((BATCH_SIZE), dtype=np.float32)

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    data_save_loop=0
    while TRAIN_DATASET.has_next_batch():
        batch_data, batch_label, lam, batch_label_b, data_original_batch, cut_rad, data_batch_a_mask, data_batch_b_mask, len_a_idx, len_b_idx, data_batch_2,\
                knn_data_batch_mixed, knn_lam, knn_data_batch_a_mask, knn_data_batch_b_mask, knn_len_a_idx, knn_len_b_idx  = TRAIN_DATASET.next_batch(augment=True, 
                                                                                convda=CONVDA, rddrop=RDDROP, 
                                                                                rsmix_prob=RSMIX_PROB, beta=BETA, 
                                                                                n_sample=N_SAMPLE, shuffle=SHUFFLE, 
                                                                                jitter=JITTER, rot=ROT, 
                                                                                rdscale=RDSCALE, shift=SHIFT)
        '''
            for saving mixed data
        '''
        if MIXED_DATA_SAVE:
            loop_dir = os.path.join(MIXED_SAVE_DIR, 'loop_'+str(data_save_loop))
            if not os.path.exists(loop_dir):
                os.mkdir(loop_dir)
            if BETA > 0 and epoch%100==0:
                for i, data_original in enumerate(data_original_batch): # original data
                    filename = 'data_{:d}_loop_{:d}_idx_{:d}_label_{:d}_original.txt'.format(epoch, data_save_loop, i, batch_label[i])
                    save_file_path = os.path.join(loop_dir,filename)
                    np.savetxt(save_file_path, data_original, fmt='%.6f', delimiter=',') 
                
                for i, data_original in enumerate(data_batch_2): # original_2 data
                    filename = 'data_{:d}_loop_{:d}_idx_{:d}_label_{:d}_original_2.txt'.format(epoch, data_save_loop, i, batch_label_b[i])
                    save_file_path = os.path.join(loop_dir,filename)
                    np.savetxt(save_file_path, data_original, fmt='%.6f', delimiter=',') 
                    
                for i, data in enumerate(batch_data): # mix data with lam, ==> part a, part b 도 보여줄 수 있음
                    filename = 'data_{:d}_loop_{:d}_idx_{:d}_label_{:d}_label_b_{:d}_radius_{:.3f}_mixed_lam_{}.txt'.format(epoch, data_save_loop, i, 
                                                                                                                    batch_label[i], batch_label_b[i], 
                                                                                                                    cut_rad, lam[i])
                    save_file_path = os.path.join(loop_dir,filename)
                    np.savetxt(save_file_path, data, fmt='%.6f', delimiter=',') 
                    
                for i, data in enumerate(data_batch_a_mask): # mask a 를 보여주어야 함 ==> part a 볼 수 잇음
                    filename = 'datamaska_{:d}_loop_{:d}_idx_{:d}_lenidx_{:d}_label_{:d}_radius_{:.3f}_mixed_lam_{}.txt'.format(epoch, data_save_loop, i, 
                                                                                                                    len_a_idx[i], batch_label[i], 
                                                                                                                    cut_rad, lam[i])
                    save_file_path = os.path.join(loop_dir,filename)
                    np.savetxt(save_file_path, data, fmt='%.6f', delimiter=',') 
                    
                for i, data in enumerate(data_batch_b_mask): # mask b 를 보여주어야 함 ==> part b 볼 수 잇음
                    filename = 'datamaskb_{:d}_loop_{:d}_idx_{:d}_lenidx_{:d}_label_b_{:d}_radius_{:.3f}_mixed_lam_{}.txt'.format(epoch, data_save_loop, i, 
                                                                                                                    len_b_idx[i], batch_label_b[i], 
                                                                                                                    cut_rad, lam[i])
                    save_file_path = os.path.join(loop_dir,filename)
                    np.savetxt(save_file_path, data, fmt='%.6f', delimiter=',') 
                    
                
                ###----KNN--------------------------------------------------------------------------------
                for i, data in enumerate(knn_data_batch_mixed): # mix data knn with lam, ==> part a, part b 도 보여줄 수 있음
                    filename = 'dataknn_{:d}_loop_{:d}_idx_{:d}_label_{:d}_label_b_{:d}_radius_{:.3f}_mixed_lam_{}.txt'.format(epoch, data_save_loop, i, 
                                                                                                                    batch_label[i], batch_label_b[i], 
                                                                                                                    cut_rad, knn_lam[i])
                    save_file_path = os.path.join(loop_dir,filename)
                    np.savetxt(save_file_path, data, fmt='%.6f', delimiter=',') 
                    
                for i, data in enumerate(knn_data_batch_a_mask): # mask a 를 보여주어야 함
                    filename = 'datamaskaknn_{:d}_loop_{:d}_idx_{:d}_lenidx_{:d}_label_{:d}_radius_{:.3f}_mixed_lam_{}.txt'.format(epoch, data_save_loop, i, 
                                                                                                                    knn_len_a_idx[i], batch_label[i], 
                                                                                                                    cut_rad, knn_lam[i])
                    save_file_path = os.path.join(loop_dir,filename)
                    np.savetxt(save_file_path, data, fmt='%.6f', delimiter=',') 
                    
                for i, data in enumerate(knn_data_batch_b_mask): # mask b 를 보여주어야 함
                    filename = 'datamaskbknn_{:d}_loop_{:d}_idx_{:d}_lenidx_{:d}_label_b_{:d}_radius_{:.3f}_mixed_lam_{}.txt'.format(epoch, data_save_loop, i, 
                                                                                                                    knn_len_b_idx[i], batch_label_b[i], 
                                                                                                                    cut_rad, knn_lam[i])
                    save_file_path = os.path.join(loop_dir,filename)
                    np.savetxt(save_file_path, data, fmt='%.6f', delimiter=',') 
                ###--------------------------------------------------------------------------------
            data_save_loop +=1
            if epoch==1:
                exit()
        '''
        '''
        #batch_data = provider.random_point_dropout(batch_data)
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label
        cur_batch_label_b[0:bsize] = batch_label_b
        cur_lam[0:bsize] = lam

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['labels_pl_b']: cur_batch_label_b,
                     ops['is_training_pl']: is_training,
                     ops['lam']: cur_lam}
        # summary, step, _, loss_val, pred_val, loss_val_a, loss_val_b, loss_val_a_lam, loss_val_b_lam = sess.run([ops['merged'], ops['step'],
            # ops['train_op'], ops['loss'], ops['pred'], loss_a, loss_b, loss_a_lam, loss_b_lam], feed_dict=feed_dict)
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        # print("loss_val_a : ",loss_val_a)
        # print("loss_val_b : ",loss_val_b)
        # print("lamda : ", lam)
        # print("loss_val_a_lam : ",loss_val_a_lam)
        # print("loss_val_b_lam : ",loss_val_b_lam)
        # print("loss_val : ", loss_val)
        # exit()
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        if (batch_idx+1)%50 == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            log_string('mean loss: %f' % (loss_sum / 50))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
        batch_idx += 1

    TRAIN_DATASET.reset()
        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,NUM_POINT,TEST_DATASET.num_channel()))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)
    cur_batch_label_b = np.zeros((BATCH_SIZE), dtype=np.int32)
    cur_lam = np.zeros((BATCH_SIZE), dtype=np.float32)
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    batch_idx = 0
    shape_ious = []
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    
    while TEST_DATASET.has_next_batch():
        # batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        batch_data, batch_label, lam, batch_label_b, _, _, _, _, _, _, _, _, _, _, _, _, _= TEST_DATASET.next_batch(augment=False)
        bsize = batch_data.shape[0]
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize,...] = batch_data
        cur_batch_label[0:bsize] = batch_label
        cur_batch_label_b[0:bsize] = batch_label_b
        cur_lam[0:bsize] = lam

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['labels_pl_b']: cur_batch_label_b,
                     ops['is_training_pl']: is_training,
                     ops['lam']: cur_lam}
        # try:
        #     summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
        #         ops['loss'], ops['pred']], feed_dict=feed_dict)
        # except:
        #     import pdb; pdb.set_trace()
        
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:bsize] == batch_label[0:bsize])
        total_correct += correct
        total_seen += bsize
        loss_sum += loss_val
        batch_idx += 1
        for i in range(0, bsize):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)        
    
    eval_accuracy = total_correct/float(total_seen)
    eval_class_accuracy = np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))
    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    log_string('eval accuracy: %f'% (eval_accuracy))
    log_string('eval avg class acc: %f' % (eval_class_accuracy))
    EPOCH_CNT += 1

    TEST_DATASET.reset()
    # return total_correct/float(total_seen)
    return eval_accuracy, eval_class_accuracy


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    start_time = time.time()
    train()
    LOG_FOUT.close()
