from random import random
import torch
import torchvision
from typing import Any, Callable, Dict, List, Optional, Tuple
import os
import numpy as np
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torchvision.utils import save_image
import random

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from math import sqrt

from models.cnn import CNN_MNIST, CNN_CIFAR10, CNN_FEMNIST, AlexNet_CIFAR10

from fedlab.utils.functional import load_dict

from fedlab.utils.dataset.slicing import random_slicing


# this class is from NIID-bench official code: 
# https://github.com/Xtra-Computing/NIID-Bench/blob/main/utils.py
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., net_id=None, total=0):
        self.std = std
        self.mean = mean
        self.net_id = net_id
        self.num = int(sqrt(total))
        if self.num * self.num < total:
            self.num = self.num + 1

    def __call__(self, tensor):
        if self.net_id is None:
            return tensor + torch.randn(tensor.size()) * self.std + self.mean
        else:
            tmp = torch.randn(tensor.size())
            filt = torch.zeros(tensor.size())
            size = int(28 / self.num)
            row = int(self.net_id / size)
            col = self.net_id % size
            for i in range(size):
                for j in range(size):
                    filt[:, row * size + i, col * size + j] = 1
            tmp = tmp * filt
            return tensor + tmp * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def get_sorted_label_index(dataset):
    labels = np.array(dataset.targets)
    idxs = np.arange(len(dataset))

    # sort sample indices according to labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

    index, label = idxs_labels[0], idxs_labels[1]
    label_index = []

    for i in range(10):
        label_index.append(index[np.where(label == i, True, False)])
        # print("label {} number {}".format(i, label_index[i].shape[0]))
    return label_index

def label_skew_parition(dataset, k=4):
    id = 0
    results = {}
    index_list = get_sorted_label_index(dataset)

    if k==10:
        for index in index_list:
            print(id)
            num_items = int(len(index)/100+1)
            for i in range(0,len(index),num_items):
                results[id] = index[i:i+num_items]
                id += 1
    
    if k==4:
        for combine in [[0,1,2],[3,4],[5,6],[7,8,9]]:
            index = []
            for i in combine:
                index += list(index_list[i])
            random.shuffle(index)
            num_items = int(len(index)/100+1)
            for i in range(0,len(index),num_items):
                results[id] = index[i:i+num_items]
                id += 1
    return results

class RotatedMNIST(data_utils.Dataset):
    def __init__(self, root, train=True, thetas=[0], d_label=0, download=True):
        self.root = os.path.expanduser(root)
        self.train = train
        self.thetas = thetas
        self.d_label = d_label
        self.download = download
        self.to_tensor = transforms.ToTensor()

        self.mnist = datasets.MNIST(self.root,
                                train=self.train,
                                download=self.download)

        self.d = np.random.choice(range(len(self.thetas)))

        self.rotated_data = []
        self.labels = self.mnist.targets
        for x, _ in self.mnist:
            d = np.random.choice(range(len(self.thetas)))
            x = self.to_tensor(transforms.functional.rotate(x, self.thetas[d]))
            self.rotated_data.append(x)

        # dir = "./datasets/augmented_mnist/degree_{}/"
        # os.mkdir(dir)
        # torch.save((self.rotated_data, self.labels),  os.path.join(dir, "data.pkl"))

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        return self.rotated_data[index], self.labels[index]
        # return self.mnist[index]

class RotatedCIFAR10(data_utils.Dataset):
    def __init__(self, root, train=True, thetas=[0], d_label=0, download=True):
        self.root = os.path.expanduser(root)
        self.train = train
        self.thetas = thetas
        self.d_label = d_label
        self.download = download
        self.to_tensor = transforms.ToTensor()

        self.cifar = datasets.CIFAR10(self.root,
                                train=self.train,
                                download=self.download)

        self.rotated_data = []
        self.labels = self.cifar.targets
        for x, _ in self.cifar:
            d = np.random.choice(range(len(self.thetas)))
            x = self.to_tensor(transforms.functional.rotate(x, self.thetas[d]))
            self.rotated_data.append(x)

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, index):
        return self.rotated_data[index], self.labels[index]

class ShiftedMNIST(data_utils.Dataset):
    def __init__(self, root, train=True, shift=0, download=True):
        self.root = os.path.expanduser(root)
        self.train = train
        self.label_shift = shift
        self.download = download
        self.mnist = datasets.MNIST(self.root,
                                train=self.train,
                                download=self.download,
                                transform=transforms.ToTensor())

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        x, y = self.mnist[index]
        return x, (y+self.label_shift)%10

class FskewedFashionMNIST(data_utils.Dataset):
    def __init__(self, root, train=True, noise_level=0, download=True) -> None:
        
        self.mnist = FashionMNIST(root=root, train=train, download=download)
        self.transform = transforms.Compose([transforms.ToTensor(), AddGaussianNoise(0., noise_level)])

        self.noise_data = []
        self.targets = self.mnist.targets
        for x,_ in self.mnist:
            x = self.transform(x)
            self.noise_data.append(x)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        return self.noise_data[index], self.targets[index]


def mnist(args):
    model = CNN_MNIST()
    trainset = torchvision.datasets.MNIST(root=args.root + "/datasets/mnist/",
                                          train=True,
                                          download=True,
                                          transform=transforms.ToTensor())

    testset = torchvision.datasets.MNIST(root=args.root + "/datasets/mnist/",
                                         train=False,
                                         download=True,
                                         transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=len(testset),
                                              drop_last=False,
                                              shuffle=False)

    if args.partition == "noniid":
        data_indices = load_dict(args.root +
                                 "/config/mnist_noniid_1000_1000.pkl")
    else:
        data_indices = load_dict(args.root + "/config/mnist_iid_100.pkl")

    return model, trainset, testset, data_indices, test_loader


def cifar10(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.root +
                                            '/datasets/cifar10/',
                                            train=True,
                                            download=True,
                                            transform=transform_train)

    testset = torchvision.datasets.CIFAR10(root=args.root +
                                           '/datasets/cifar10/',
                                           train=False,
                                           download=True,
                                           transform=transform_test)

    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=len(testset),
                                              drop_last=False,
                                              shuffle=False)

    #model = AlexNet_CIFAR10()
    model = CNN_CIFAR10()
    if args.partition == "noniid":
        data_indices = load_dict(args.root +
                                 "/config/cifar10_noniid_100_200.pkl")
    else:
        data_indices = load_dict(args.root + "/config/cifar10_iid_100.pkl")

    return model, trainset, testset, data_indices, test_loader


def femnist(args):
    model = CNN_FEMNIST()
    train_transform, test_transform = get_data_transform('mnist')
    #train_dataset = FEMNIST('/data/zengdun/dataset/data/femnist', dataset='train', transform=train_transform)
    test_dataset = FEMNIST('/data/zengdun/dataset/data/femnist',
                           dataset='test',
                           transform=test_transform)
    #val_dataset = FEMNIST('/data/zengdun/dataset/data/femnist', dataset='val', transform=test_transform)

    test_loader = DataLoader(test_dataset, batch_size=512)

    return model, test_loader
