#!/usr/bin/env python3

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset

import os
import sys
import math
import shutil
# import setproctitle
import scipy.io
import numpy as np
import pdb

#import densenet
import architecture
import dataloader_mnist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=128)
    parser.add_argument('--num_models', type=int, default=10)
    parser.add_argument('--model', default='baseline')
    parser.add_argument('--setname', default='pairwise')
    parser.add_argument('--newdata', default=False)
    parser.add_argument('--nEpochs', type=int, default=5)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save', default='model')
    parser.add_argument('--nTokens', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not os.path.exists('model'):
        os.mkdir('model')

    # use the args.model + args.setname as the unique experimentID
    args.save = 'model/%s-%s'%(args.model, args.setname)
    if not os.path.exists(args.save):
        os.mkdir(args.save)

    # load the data loader
    data_mnist = dataloader_mnist.MNIST('data', download=True)
    trainLoader, testLoader, class_labels, oracles = data_mnist.generate_split(batch_size=args.batchSz, setname=args.setname, newdata=args.newdata)
    args.nClasses = len(class_labels)

    # train multiple models
    err_models = []
    for i in range(args.num_models):
        args.seed = i
        print('training model %d' % i)
        err_train, err_test = train_net(args, trainLoader, testLoader)
        err_models.append((err_train, err_test))

    # bookmaking the all the errors

    output = ['id\terr_test\terr_train']
    with open(args.save + '/errors.txt','w') as f:
        for i, item in enumerate(err_models):
            output.append('%d\t%.2f\t%.2f'%(i, item[1], item[0]))
        f.write('\n'.join(output))
    print '\n'.join(output)

def train_net(args, trainLoader, testLoader):
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.model == 'baseline':
        # the model is 3 conv layer + GAP at the end
        net = architecture.Baseline(nTokens=args.nTokens, nClasses=args.nClasses)

    #net = densenet.DenseNet(growthRate=3, depth=10, reduction=0.5,
    #                        bottleneck=True, nClasses=args.nClasses)
    #print('  + Number of params: {}'.format(sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    trainF = open(os.path.join(args.save, '%d_train.csv'%args.seed), 'w')
    testF = open(os.path.join(args.save, '%d_test.csv'%args.seed), 'w')

    err_best = 1000
    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        err_train = train(args, epoch, net, trainLoader, optimizer, trainF)
        err_test = test(args, epoch, net, testLoader, optimizer, testF)
        torch.save(net, os.path.join(args.save, '%d_modellatest.pth'%args.seed))
        if err_test<err_best:
            torch.save(net, os.path.join(args.save,'%d_modelbest.pth'%args.seed))
            err_best = err_test

    trainF.close()
    testF.close()
    return err_train, err_best

def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    count_batch = 200
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        if batch_idx % count_batch == 0:
            print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()
    return err

def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()
    return err

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 80: lr = 1e-1
        elif epoch == 120: lr = 1e-2
        elif epoch == 160: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__=='__main__':
    main()
