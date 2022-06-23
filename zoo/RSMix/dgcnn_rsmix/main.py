#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RSMix:
@Author: Dogyoon Lee
@Contact: dogyoonlee@gmail.com
@File: main.py
@Time: 2020/11/23 13:46 PM

DGCNN:
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM
"""


from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
from model import PointNet, DGCNN
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import time
from datetime import datetime
import provider
import rsmix_provider
from ModelNetDataLoader import ModelNetDataLoader
from modelnetc_utils import eval_corrupt_wrapper, ModelNetC


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def train(args, io):
    if args.modelnet10:
        TRAIN_DATASET = ModelNetDataLoader(root=args.data_path, npoint=args.num_points, split='train', normal_channel=args.normal, modelnet10=True)
        TEST_DATASET = ModelNetDataLoader(root=args.data_path, npoint=args.num_points, split='test', normal_channel=args.normal, modelnet10=True)
        train_loader = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True)
        test_loader = DataLoader(TEST_DATASET, batch_size=args.test_batch_size, shuffle=False, num_workers=8, drop_last=False)
        num_class = 10

    else:
        train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=8,
                                batch_size=args.batch_size, shuffle=True, drop_last=True) 
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                                 batch_size=args.test_batch_size, shuffle=True, drop_last=False)
        num_class = 40
    # drop last : don't use the last batch if the size of it is different to the other ones(when its True)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    if args.model == 'pointnet':
        model = PointNet(args).to(device)
    elif args.model == 'dgcnn':
        model = DGCNN(args, output_channels=num_class).to(device)
    else:
        raise Exception("Not implemented")
    print(str(model))

    model = nn.DataParallel(model) # for multi-gpu
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=1e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
    
    criterion = cal_loss

    best_test_acc = 0
    best_avg_class_acc = 0
    conv_epoch = 0
    for epoch in range(args.epochs):
        log_string(str(datetime.now()))
        log_string('**** EPOCH %03d ****' % (epoch))
        scheduler.step()
        ####################
        # Train
        ####################
        train_loss = 0.0
        count = 0.0
        model.train()
        train_pred = []
        train_true = []
        for data, label in train_loader:
            '''
            implement augmentation
            '''
            rsmix = False
            # for new augmentation code, remove squeeze because it will be applied after augmentation.
            # default from baseline model, scale, shift, shuffle was default augmentation
            if args.rot or args.rdscale or args.shift or args.jitter or args.shuffle or args.rddrop or (args.beta is not 0.0):
                data = data.cpu().numpy()
            if args.rot:
                data = provider.rotate_point_cloud(data)
                data = provider.rotate_perturbation_point_cloud(data)
            if args.rdscale:
                tmp_data = provider.random_scale_point_cloud(data[:,:,0:3])
                data[:,:,0:3] = tmp_data
            if args.shift:
                tmp_data = provider.shift_point_cloud(data[:,:,0:3])
                data[:,:,0:3] = tmp_data
            if args.jitter:
                tmp_data = provider.jitter_point_cloud(data[:,:,0:3])
                data[:,:,0:3] = tmp_data
            if args.rddrop:
                data = provider.random_point_dropout(data)
            if args.shuffle:
                data = provider.shuffle_points(data)
            r = np.random.rand(1)
            if args.beta > 0 and r < args.rsmix_prob:
                rsmix = True
                data, lam, label, label_b = rsmix_provider.rsmix(data, label, beta=args.beta, n_sample=args.nsample, KNN=args.knn)
            if args.rot or args.rdscale or args.shift or args.jitter or args.shuffle or args.rddrop or (args.beta is not 0.0):
                data = torch.FloatTensor(data)
            if rsmix:
                lam = torch.FloatTensor(lam)
                lam, label_b = lam.to(device), label_b.to(device).squeeze()
            data, label = data.to(device), label.to(device).squeeze()

            if rsmix:
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                opt.zero_grad()
                logits = model(data)

                loss = 0
                for i in range(batch_size):
                    loss_tmp = criterion(logits[i].unsqueeze(0), label[i].unsqueeze(0).long())*(1-lam[i]) \
                        + criterion(logits[i].unsqueeze(0), label_b[i].unsqueeze(0).long())*lam[i]
                    loss += loss_tmp
                loss = loss/batch_size

            else:
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                opt.zero_grad()
                logits = model(data)
                loss = criterion(logits, label)
            
            '''
            from above to here
            '''
            # data = data.permute(0, 2, 1)
            # batch_size = data.size()[0]
            # opt.zero_grad()
            # logits = model(data)
            # loss = criterion(logits, label)
            
            loss.backward()
            opt.step()
            preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        outstr = 'Train epoch %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                 train_loss*1.0/count,
                                                                                 train_acc,
                                                                                 metrics.balanced_accuracy_score(
                                                                                     train_true, train_pred))
        io.cprint(outstr)
        

        LOG_FOUT.write(outstr+'\n')
        LOG_FOUT.flush()

        ####################
        # Test
        ####################
        log_string('---- EPOCH %03d EVALUATION ----'%(epoch))

        test_loss = 0.0
        count = 0.0
        model.eval()
        test_pred = []
        test_true = []
        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            batch_size = data.size()[0]
            logits = model(data)
            loss = criterion(logits, label)
            preds = logits.max(dim=1)[1]
            count += batch_size
            test_loss += loss.item() * batch_size
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
        outstr = 'Test epoch %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch,
                                                                              test_loss*1.0/count,
                                                                              test_acc,
                                                                              avg_per_class_acc)
        io.cprint(outstr)

        LOG_FOUT.write(outstr+'\n')
        LOG_FOUT.flush()

        if test_acc >= best_test_acc:
            best_test_acc = test_acc
            conv_epoch = epoch
            torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
            log_string('Model saved in file : checkpoints/%s/models/model.t7' %(args.exp_name))
        # if avg_per_class_acc >= best_avg_class_acc:
            best_avg_class_acc = avg_per_class_acc
            # torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
            # log_string('Model class_acc saved in file : checkpoints/%s/models/model_class_acc.t7' %(args.exp_name))
        log_string('*** best accuracy *** - %f' %(best_test_acc))
        log_string('*** at then, best class accuracy *** -  %f' %(best_avg_class_acc))
    
    execution_time = time.time()-start_time
    hour = execution_time//3600
    minute = (execution_time-hour*3600)//60
    second = execution_time-hour*3600-minute*60
    log_string('... End of the Training ...')
    log_string("trainig time : %.2f sec, %d min, %d hour" %(float(second), int(minute), int(hour)))
    log_string('*** training accuracy when best accuracy *** - %f' %(train_acc))
    log_string('*** best accuracy *** - %f' %(best_test_acc))
    log_string('*** at then, best class accuracy *** -  %f' %(best_avg_class_acc))
    log_string('*** conv epoch *** - %d' %(conv_epoch))

def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                             batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    #Try to load models
    model = DGCNN(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_acc = 0.0
    count = 0.0
    test_true = []
    test_pred = []
    for data, label in test_loader:

        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        batch_size = data.size()[0]
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--eval_corrupt', type=bool, default=False,
                        help='evaluate the model under corruption')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    # added arguments
    parser.add_argument('--rdscale', action='store_true', help='random scaling data augmentation')
    parser.add_argument('--shift', action='store_true', help='random shift data augmentation')
    parser.add_argument('--shuffle', action='store_true', help='random shuffle data augmentation')
    parser.add_argument('--rot', action='store_true', help='random rotation augmentation')
    parser.add_argument('--jitter', action='store_true', help='jitter augmentation')
    parser.add_argument('--rddrop', action='store_true', help='random point drop data augmentation')
    parser.add_argument('--rsmix_prob', type=float, default=0.5, help='rsmix probability')
    parser.add_argument('--beta', type=float, default=0.0, help='scalar value for beta function')
    parser.add_argument('--nsample', type=float, default=512, help='default max sample number of the erased or added points in rsmix')
    parser.add_argument('--modelnet10', action='store_true', help='use modelnet10')
    parser.add_argument('--normal', action='store_true', help='use normal')
    parser.add_argument('--knn', action='store_true', help='use knn instead ball-query function')
    parser.add_argument('--data_path', type=str, default='./data/modelnet40_normal_resampled', help='dataset path')
    
    
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    if not os.path.exists('./log'): os.mkdir('./log')
    LOG_DIR = os.path.join('./log',args.exp_name)
    if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
    LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
    LOG_FOUT.write(str(args)+'\n')

    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    start_time = time.time()

    if not args.eval and not args.eval_corrupt:
        train(args, io)
    elif args.eval:
        test(args, io)
    elif args.eval_corrupt:
        device = torch.device("cuda" if args.cuda else "cpu")
        model = DGCNN(args).to(device)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(args.model_path))
        model = model.eval()

        def test_corrupt(args, split, model):
            test_loader = DataLoader(ModelNetC(split=split),
                                     batch_size=args.test_batch_size, shuffle=True, drop_last=False)
            test_true = []
            test_pred = []
            for data, label in test_loader:
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                logits = model(data)
                preds = logits.max(dim=1)[1]
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            return {'acc': test_acc, 'avg_per_class_acc': avg_per_class_acc}
        eval_corrupt_wrapper(model, test_corrupt, {'args': args})
    
    LOG_FOUT.close()

