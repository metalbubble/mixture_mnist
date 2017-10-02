# visualize the tokens for the given model
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

import scipy.io
import numpy as np
import pdb
from PIL import Image
import os
import cv2
import dataloader_mnist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='baseline')
    parser.add_argument('--setname', default='pairwise')
    parser.add_argument('--num_models', type=int, default=10)
    parser.add_argument('--batchSz', type=int, default=128)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--nClasses', type=int, default=45)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output', default='')
    parser.add_argument('--prefix', default='')
    parser.add_argument('--class_labels', default='')

    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    data_mnist = dataloader_mnist.MNIST('data')
    print('Loading %s' % args.setname)
    _, testLoader, class_labels, oracles = data_mnist.generate_split(
            batch_size = args.batchSz, setname=args.setname)
    args.nClasses = len(class_labels)
    args.class_labels = class_labels
    oracle_test = oracles[1] # the first is the oracle for train, and second is the oracle for test.
    print('Finished loading %s' % args.setname)
    if args.prefix != '':
        args.setname = args.setname + '-' + args.prefix

    for i in range(args.num_models):
        print 'testing model:%s %s %d' %(args.model, args.setname, i)
        modelfile = 'model/%s-%s/%d_modelbest.pth' % (args.model,args.setname, i)
        if not os.path.isfile(modelfile):
            print('no file:', modelfile)
            continue
        net = torch.load(modelfile)
        args.output = 'output/%s-%s/%03d'%(args.model, args.setname, i)
        root_folder = 'output/%s-%s' % (args.model, args.setname)

        if not os.path.exists('output'):
            os.mkdir('output')
        if not os.path.exists(root_folder):
            os.mkdir(root_folder)

        if args.cuda:
            net.cuda()
        net.eval()
        visualize(args, net, testLoader, oracle_test)

def test(args, net, testLoader):
    # test the accuracy
    net.eval()
    test_loss = 0
    incorrect = 0
    err_class = np.zeros(args.nClasses)
    conf_mat = np.zeros((args.nClasses, args.nClasses), dtype=np.int32)
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        test_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        miss = pred.ne(target.data).cpu()
        incorrect += miss.sum()
        x = pred.cpu().numpy() + args.nClasses * target.data.cpu().numpy()
        bincount_2d = np.bincount(x.astype(np.int32), minlength=args.nClasses ** 2)
        conf_mat += bincount_2d.reshape((args.nClasses, args.nClasses))


    test_loss = test_loss
    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))
    conf_mat = conf_mat.astype(np.float32)
    return conf_mat / conf_mat.sum(1).clip(min=1e-12)[:, None]

def visualize(args, net, testLoader, oracle_test):
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.cpu().numpy()) # (num_class, num_unit)


    print('running test')
    confusion_matrix = test(args, net, testLoader)
    acc_class = confusion_matrix.diagonal()
    print('extracting features')
    features_set, images_test, labels = extractfeature(args, net, testLoader)
    features = features_set[0]
    # step1: get the max idx for each unit
    # step2: segment the images
    num_samples, num_tokens, height, weight = features.shape
    maxvalues_all = np.max(np.max(features,3),2) # (num_sample, num_units)
    montage_units = []
    heatmap_units = []
    pr_units = []
    for unitID in range(maxvalues_all.shape[1]):
        print 'processing unit%d' % unitID
        maxvalues_unit = maxvalues_all[:, unitID]
        pr_threshs = compute_precision_recall(maxvalues_unit, oracle_test) # compute the precision and recall
        features_unit = np.squeeze(features[:, unitID, :, :])
        montage_unit, heatmap_unit = generate_unitmap(images_test, features_unit, maxvalues_unit)
        montage_units.append(montage_unit)
        heatmap_units.append(heatmap_unit)
        pr_units.append(pr_threshs)
    output_result(args, montage_units, heatmap_units, pr_units, acc_class, weight_softmax)

def output_result(args, montage_units, heatmap_units, pr_units, acc_class, weight_softmax):
    # output the result to HTML
    html_file = '%s.html' % args.output
    directory_unit = '%s' % args.output
    if not os.path.exists(directory_unit):
        os.mkdir(directory_unit)

    output_lines = []
    values_sort = []
    for unitID, montage_unit in enumerate(montage_units):
        recall10 = pr_units[unitID][0][1]
        recall100 = pr_units[unitID][1][1]
        idx_digit_10 = np.argsort(recall10) # this is the ascending order
        idx_digit_10 = idx_digit_10[::-1]

        idx_digit_100 = np.argsort(recall100) # this is the ascending order
        idx_digit_100 = idx_digit_100[::-1]

        values_sort.append(recall10[idx_digit_10[0]])
        filename_unit = '%s/unit%02d.jpg' % (args.output, unitID)
        filename_mask = '%s/mask_unit%02d.jpg' % (args.output, unitID)
        link_unit = '/'.join(filename_unit.split('/')[2:])
        link_mask = '/'.join(filename_mask.split('/')[2:])

        output_lines.append('<p>Unit%02d for detecting digit %d with recall@10=%.2f and recall@100=%.2f<br>Top activated images:<br><img src="%s"><br>Activation:<br><img src="%s"></p>'%(unitID, idx_digit_100[0], recall10[idx_digit_100[0]], recall100[idx_digit_100[0]], link_unit, link_mask))
        # output montage of top ranked images
        montage_unit_output = np.transpose(montage_unit, (1,2,0))
        montage_unit_output = np.squeeze(montage_unit_output)
        img = Image.fromarray(montage_unit_output)
        img.save(filename_unit)
        # output the heatmap
        montage_unit_output = np.transpose(heatmap_units[unitID], (1,2,0))
        montage_unit_output = np.squeeze(montage_unit_output)
        img = Image.fromarray(montage_unit_output)
        img.save(filename_mask)

    # sort the unit by their precision
    idx_sorted = np.argsort(values_sort)
    print html_file
    with open(html_file,'w') as f:
        # output the class-linked units
        output_classlink = []
        for i in range(weight_softmax.shape[0]):
            weight_class = weight_softmax[i]
            idx_sorted_units = np.argsort(weight_class)[::-1]
            top_line = ['Unit%02d (%.2f)'%(idx_sorted_units[j], weight_class[idx_sorted_units[j]]) for j in range(3)]
            output_classlink.append(top_line)

        # output the class accuracy and their most contributed units
        f.write('<h4>Class Accuracy:%.2f</h4>'%np.mean(acc_class))
        idx_sorted_class = np.argsort(acc_class)
        output_acc = ['<strong>Class%02d</strong> (%s):acc=%.2f, %s'%(i, np.array2string(args.class_labels[i])[1:-1], acc_class[i], ','.join(output_classlink[i])) for i in idx_sorted_class]

        # not plot all of them
        if len(output_acc) > 15:
            interval = int(len(output_acc)/15)
            output_acc = output_acc[::interval]
        f.write('<br>'.join(output_acc))
        # output the unit precision and visualization
        f.write('<h4>Unit Precision and Visualization</h4>')
        for i in range(len(output_lines)):
            f.write('\n')
            f.write(output_lines[i])

    #print 'http://places.csail.mit.edu/deepscene/small-projects/mixture_mnist/' + html_file

def compute_precision_recall(responses_unit, oracle_test):
    threshs = [10,100] # threshold for the recall
    idx_sorted = np.argsort(responses_unit) # this is the ascending order
    idx_sorted = idx_sorted[::-1]
    gt_sorted = oracle_test[idx_sorted, :]
    totals = np.sum(gt_sorted, axis=0) # the total number of gt in each bin
    pr_threshs = []

    for t in threshs:
        recall = np.sum(gt_sorted[:t,:], axis=0) *1.0 / t
        precision = np.sum(gt_sorted[:t,:], axis=0) * 1.0 / totals
        pr_threshs.append((precision, recall))
    return pr_threshs # return the threshold at 100

def generate_unitmap(images, features_unit, maxvalues_unit):
    idx_sorted = np.argsort(maxvalues_unit) # this is the ascending order
    idx_sorted = idx_sorted[::-1]
    num_top = 20
    images_top = images[idx_sorted[:num_top]]
    features_top = features_unit[idx_sorted[:num_top]]
    montage, montage_mask = create_montage(images_top, features_top, column_size=20)
    return montage, montage_mask
    # create montage of the images

def extractfeature(args, net, testLoader):
    net.eval()
    features_names = ['features']
    features_blobs = []
    def hook_feature(module, input, output):
        # hook the feature extractor
        features_blobs.append(np.squeeze(output.data.cpu().numpy()))

    for name in features_names:
        net._modules.get(name).register_forward_hook(hook_feature)

    num_samples = len(testLoader.dataset)

    features_results = [None] * len(features_names)
    labels_results = np.zeros(num_samples)
    images_results = []
    for batch_idx, (data, target) in enumerate(testLoader):
        del features_blobs[:]
        images_results.append(data.numpy())
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        if features_results[0] is None:
            # initialize the feature variable
            for i, feat_batch in enumerate(features_blobs):
                size_features = ()
                size_features = size_features + (num_samples,)
                size_features = size_features + feat_batch.shape[1:]
                features_results[i] = np.zeros(size_features)
                #print features_results[i].shape
        start_idx = batch_idx*args.batchSz
        end_idx = min((batch_idx+1)*args.batchSz, num_samples)
        labels_results[start_idx:end_idx] = target.data.cpu().numpy()
        for i, feat_batch in enumerate(features_blobs):
            features_results[i][start_idx:end_idx] = feat_batch

    return features_results, np.concatenate(images_results, axis=0), labels_results

def create_montage(images,features, column_size=10, margin=2):
    # a function to create a montage of images
    if len(images.shape) == 3:
        # if grey image, add a singlex channel
        images = images.reshape(images.shape[0], 1, images.shape[1], images.shape[2])

    num_images, num_channel, height_img, width_img = images.shape
    width = (width_img + margin)*column_size
    row_size = np.ceil(num_images * 1.0 / column_size)
    height = int((height_img + margin)*row_size)
    montage = np.ones([num_channel, height, width], dtype=np.uint8)
    montage[:] = 255
    montage_mask = np.ones([3, height, width], dtype=np.uint8)
    montage_mask[:] = 255
    max_x = 0
    max_y = 0
    offset_x = 0
    offset_y = 0
    for i in range(num_images):
        image = images[i]
        # normalize and resize the feature mask
        mask = features[i]
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        mask_img = np.uint8(mask * 255)
        mask_img = cv2.resize(mask_img, (height_img, width_img))
        heatmap = cv2.applyColorMap(mask_img, cv2.COLORMAP_HOT)
        montage[:, offset_y:offset_y + height_img, offset_x:offset_x + width_img] = image
        montage_mask[:, offset_y:offset_y + height_img, offset_x:offset_x + width_img] = np.transpose(heatmap, (2,0,1))

        max_x = max(max_x, offset_x + width_img)
        max_y = max(max_y, offset_y + height_img)

        if i % column_size == column_size-1:
            offset_y = max_y + margin
            offset_x = 0
        else:
            offset_x += margin + width_img
    return montage, montage_mask
main()
