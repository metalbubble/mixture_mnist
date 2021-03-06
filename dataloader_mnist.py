# the data converter of minist

from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import torch
import codecs
import cv2
import numpy as np
import pdb
import random
from torch.utils.data import DataLoader, TensorDataset

class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def dataset_classes(self, setname):
        if setname == 'pairwise':
            # permute the two digits, so totally 45 classes
            class_labels = []
            for i in range(9):
                for j in range(i+1, 10):
                    class_labels.append([(i,j)])
            return class_labels

        if setname == 'xor':
            return [[(0, 0), (1, 1)], [(0, 1)]]
        if setname == 'greater':
            return [
              [(0,0)],
              [(1,1),(1,0)],
              [(2,2),(2,1),(2,0)],
              [(3,3),(3,2),(3,1),(3,0)],
              [(4,4),(4,3),(4,2),(4,1),(4,0)],
              [(5,5),(5,4),(5,3),(5,2),(5,1),(5,0)],
              [(6,6),(6,5),(6,4),(6,3),(6,2),(6,1),(6,0)],
              [(7,7),(7,6),(7,5),(7,4),(7,3),(7,2),(7,1),(7,0)],
              [(8,8),(8,7),(8,6),(8,5),(8,4),(8,3),(8,2),(8,1),(8,0)],
              [(9,9),(9,8),(9,7),(9,6),(9,5),(9,4),(9,3),(9,2),(9,1),(9,0)]]
        if setname == 'sum':
            return [
              [(0,0)],
              [(1,0)],
              [(2,0),(1,1)],
              [(3,0),(2,1)],
              [(4,0),(3,1),(2,2)],
              [(5,0),(4,1),(3,2)],
              [(6,0),(5,1),(4,2),(3,3)],
              [(7,0),(6,1),(5,2),(4,3)],
              [(8,0),(7,1),(6,2),(5,3),(4,4)],
              [(9,0),(8,1),(7,2),(6,3),(5,4)],
              [(9,1),(8,2),(7,3),(6,4),(5,5)],
              [(9,2),(8,3),(7,4),(6,5)],
              [(9,3),(8,4),(7,5),(6,6)],
              [(9,4),(8,5),(7,6)],
              [(9,5),(8,6),(7,7)],
              [(9,6),(8,7)],
              [(9,7),(8,8)],
              [(9,8)],
              [(9,9)]]
        if setname == 'redundancy':
            # generate the split with redundant digits
            return 0
        if setname == '10digits':
            return [[(0)],[(1)],[(2)],[(3)],[(4)],[(5)],[(6)],[(7)],[(8)],[(9)]]
            # baseline: there will be only one digit in each sample
        assert False, "Unrecognized setname %s" % setname

    def generate_trainval(self, setname=None, newdata=False):
        filename = 'data/split_%s_digit.npz' % setname
        if newdata or not os.path.exists(filename):
            print('generate new training data for %s split...' % setname)
            # this will determine the combination of classes
            class_labels = self.dataset_classes(setname)
            images_train, labels_train, oracle_train, class_labels = self.return_pairwise_digits(
                    class_labels=class_labels)
            images_test,  labels_test, oracle_test, _  = self.return_pairwise_digits(train=0,
                    class_labels=class_labels)
            np.savez(filename,
                    images_train=images_train,
                    labels_train=labels_train,
                    images_test=images_test,
                    labels_test=labels_test,
                    oracle_test=oracle_test,
                    oracle_train=oracle_train,
                    class_labels=class_labels)
        else:
            print('load exisiting %s split from %s' % (setname, filename))
            npzfile = np.load(filename)
            images_train = npzfile['images_train']
            labels_train = npzfile['labels_train']
            images_test = npzfile['images_test']
            labels_test = npzfile['labels_test']
            class_labels = npzfile['class_labels']
            oracle_train = npzfile['oracle_train']
            oracle_test = npzfile['oracle_test']
        return images_train, images_test, labels_train, labels_test, oracle_train, oracle_test, class_labels

    def return_pairwise_digits(self,  class_labels, train=1):
        # generate the pairwise samples: total number of classes = 45
        size_digit = (12, 12)
        size_canvas = (32, 32)
        range_rand = size_canvas[0] - size_digit[0]

        if train == 1:
            num_digit_perclass = 10000
            data_images = self.train_data.numpy()
            data_labels = self.train_labels.numpy()
        else:
            num_digit_perclass = 100
            data_images = self.test_data.numpy()
            data_labels = self.test_labels.numpy()
        indices_digit = []

        for i in range(10):
            indices_digit.append(np.squeeze(np.where(data_labels == i)))

        data_mixture = []
        labels_mixture = []
        oracle_mixture = [] # this variable contains the digits shown in each sample, for evaluating the precision and recall of unit
        for classIDX  in range(len(class_labels)):
            curLabels = np.ones(num_digit_perclass, dtype=np.uint8) * classIDX
            labels_mixture.append(curLabels)
            canvas_class = np.zeros((num_digit_perclass, size_canvas[0], size_canvas[1]), dtype=np.uint8)
            oracle_class = np.zeros((num_digit_perclass, 10), dtype=np.uint8) # one hot vector for 10 digits
            for i in range(num_digit_perclass):
                # create one mixture sample by adding any number of digits random-spatially
                curIndice = random.choice(class_labels[classIDX])
                if isinstance(curIndice, int):
                    curIndice = [curIndice]

                canvas_sample = np.zeros((size_canvas[0], size_canvas[1]), dtype=np.uint8)
                for idx in curIndice:
                    randIDX = np.int64(np.ceil(np.random.rand(2) * range_rand)) # the random spatial location
                    oracle_class[i,idx] = 1
                    IDX_rand_sample = np.random.choice(indices_digit[idx])
                    sample = data_images[IDX_rand_sample]
                    sample = cv2.resize(sample, size_digit)
                    canvas_sample[randIDX[0]:randIDX[0]+size_digit[0], randIDX[1]:randIDX[1]+size_digit[1]] = canvas_sample[randIDX[0]:randIDX[0]+size_digit[0], randIDX[1]:randIDX[1]+size_digit[1]]+sample
                    canvas_class[i] = canvas_sample

                canvas_sample[canvas_sample>255] = 255
            data_mixture.append(canvas_class)
            oracle_mixture.append(oracle_class)
        data_mixture = np.concatenate(data_mixture, axis=0)
        labels_mixture = np.concatenate(labels_mixture, axis=0)
        oracle_mixture = np.concatenate(oracle_mixture, axis=0)
        # shuffle the samples
        shuffleIDX = np.random.permutation(labels_mixture.shape[0])
        data_mixture = data_mixture[shuffleIDX,]
        labels_mixture = labels_mixture[shuffleIDX,]
        oracle_mixture = oracle_mixture[shuffleIDX,]
        return data_mixture, labels_mixture, oracle_mixture, class_labels

    def generate_split(self, batch_size=128, setname='pairwise', newdata=False):
        # generate the data split from the original MNIST data
        images_train, images_test, labels_train, labels_test, oracle_train, oracle_test, class_labels  = self.generate_trainval(setname=setname, newdata=newdata)
        images_train = images_train.astype(np.float32)
        images_test = images_test.astype(np.float32)
        images_train = images_train.reshape((images_train.shape[0], 1, images_train.shape[1], images_train.shape[2]))
        images_test = images_test.reshape((images_test.shape[0], 1, images_test.shape[1], images_test.shape[2]))
        # create the train data loader and val data loader
        train_d = TensorDataset(torch.from_numpy(images_train), torch.from_numpy(labels_train))
        trainLoader = DataLoader(train_d, batch_size = batch_size, shuffle=True)
        val_d = TensorDataset(torch.from_numpy(images_test), torch.from_numpy(labels_test))
        testLoader = DataLoader(val_d, batch_size = batch_size, shuffle=False)
        return trainLoader, testLoader, class_labels, (oracle_train, oracle_test)



class FashionMNIST(MNIST):
    """`Fashion MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.
    """
    urls = [
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
        'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
    ]


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def parse_byte(b):
    if isinstance(b, str):
        return ord(b)
    return b


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        labels = [parse_byte(b) for b in data[8:]]
        assert len(labels) == length
        return torch.LongTensor(labels)


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        idx = 16
        for l in range(length):
            img = []
            images.append(img)
            for r in range(num_rows):
                row = []
                img.append(row)
                for c in range(num_cols):
                    row.append(parse_byte(data[idx]))
                    idx += 1
        assert len(images) == length
        return torch.ByteTensor(images).view(-1, 28, 28)

