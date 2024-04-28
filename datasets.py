import os
from torchvision import datasets, transforms
import torch.utils.data as data_utils
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from fedlab.utils.dataset.slicing import random_slicing
from settings import label_skew_parition

class BaseDataset(data_utils.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

class HybridMNISTPartitioner():
    def __init__(self, root, save_dir, train=True) -> None:
        self.root = os.path.expanduser(root)
        self.dir = save_dir 
        # "./datasets/rotated_mnist/"
        if os.path.exists(save_dir) is not True:
            os.mkdir(save_dir)
            os.mkdir(os.path.join(save_dir, "train"))
            os.mkdir(os.path.join(save_dir, "test"))

    def pre_process(self):
        # train
        mnist = torchvision.datasets.MNIST("./datasets/mnist/", train=True, transform=transforms.ToTensor(), download=True)
        fmnist = torchvision.datasets.FashionMNIST("./datasets/fmnist/", train=True, transform=transforms.ToTensor(), download=True)

        samples, labels = [], []
        for x, y in mnist:
            samples.append(x)
            labels.append(y)
        mnist_data_indices = random_slicing(mnist, num_clients=100)
        for id, indices in mnist_data_indices.items():
            data, label = [], []
            for idx in indices:
                x, y = samples[idx], labels[idx]
                data.append(x)
                label.append(y)
            dataset = BaseDataset(data, label)
            torch.save(dataset, os.path.join(self.dir, "train", "data{}.pkl".format(id)))

        samples, labels = [], []
        data_indices = random_slicing(fmnist, num_clients=100)
        for x, y in fmnist:
            samples.append(x)
            labels.append(y)
        for id, indices in data_indices.items():
            data, label = [], []
            for idx in indices:
                x, y = samples[idx], labels[idx]
                data.append(x)
                label.append(y)
            dataset = BaseDataset(data, label)
            torch.save(dataset, os.path.join(self.dir, "train", "data{}.pkl".format(id+len(mnist_data_indices))))

    def get_dataset(self, id, type="train"):
        if type=="test":
            if id < 100:
                dataset = torchvision.datasets.MNIST("./datasets/mnist/", train=False, transform=transforms.ToTensor())
            if id >= 100:
                dataset = torchvision.datasets.FashionMNIST("./datasets/fmnist/", train=False, transform=transforms.ToTensor())
        else:
            dataset = torch.load(os.path.join(self.dir, type, "data{}.pkl".format(id)))        
        return dataset

    def get_data_loader(self, id, batch_size=None, type="train"):
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader

class ComplexMNISTPartitioner():
    def __init__(self, root, save_dir, train=True):
        self.root = os.path.expanduser(root)
        self.dir = save_dir 
        # "./datasets/rotated_mnist/"
        if os.path.exists(save_dir) is not True:
            os.mkdir(save_dir)
            os.mkdir(os.path.join(save_dir, "train"))
            os.mkdir(os.path.join(save_dir, "test"))

    def pre_process(self):
        # train
        mnist = datasets.MNIST(self.root, train=True)
        to_tensor = transforms.ToTensor()

        samples, labels = [], []
        for x, y in mnist:
            samples.append(to_tensor(x))
            labels.append(y)

        data_indices = label_skew_parition(mnist, slice=50)
        print(len(data_indices))
        for id, indices in data_indices.items():
            data, label = [], []
            for idx in indices:
                x, y = samples[idx], labels[idx]
                data.append(x)
                label.append(y)
            dataset = BaseDataset(data, label)
            torch.save(dataset, os.path.join(self.dir, "train", "data{}.pkl".format(id)))

        samples, labels = [], []
        for x, y in mnist:
            samples.append(to_tensor(transforms.functional.rotate(x, 180)))
            labels.append(y)
        for id, indices in data_indices.items():
            data, label = [], []
            for idx in indices:
                x, y = samples[idx], labels[idx]
                data.append(x)
                label.append(y)
            dataset = BaseDataset(data, label)
            torch.save(dataset, os.path.join(self.dir, "train", "data{}.pkl".format(id+len(data_indices))))

    def get_dataset(self, id, type="train"):
        dataset = torch.load(os.path.join(self.dir, type, "data{}.pkl".format(id)))
        return dataset

    def get_data_loader(self, id, batch_size=None, type="train"):
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader

class RotatedMNIST():
    def __init__(self, root, save_dir, train=True):
        self.root = os.path.expanduser(root)
        self.dir = save_dir 
        # "./datasets/rotated_mnist/"
        if os.path.exists(save_dir) is not True:
            os.mkdir(save_dir)
            os.mkdir(os.path.join(save_dir, "train"))
            os.mkdir(os.path.join(save_dir, "test"))
        
    def pre_process(self, thetas = [0, 90, 180, 270], shards=100):
        # train
        mnist = datasets.MNIST(self.root, train=True)
        cid = 0
        to_tensor = transforms.ToTensor()
        for theta in thetas:
            rotated_data = []
            labels = []
            partition = random_slicing(mnist, shards)
            for x, y in mnist:
                x = to_tensor(transforms.functional.rotate(x, theta))
                rotated_data.append(x)
                labels.append(y)
            for key, value in partition.items():
                data = [rotated_data[i] for i in value]
                label = [labels[i] for i in value]
                dataset = BaseDataset(data, label)
                torch.save(dataset, os.path.join(self.dir, "train" ,"data{}.pkl".format(cid)))
                cid += 1

        # test
        mnist_test = datasets.MNIST(self.root, train=False)
        labels = mnist_test.targets
        for i, theta in enumerate(thetas):
            rotated_data = []
            for x, y in mnist_test:
                x = to_tensor(transforms.functional.rotate(x, theta))
                rotated_data.append(x)
            dataset = BaseDataset(rotated_data, labels)
            torch.save(dataset, os.path.join(self.dir,"test", "data{}.pkl".format(i)))

    def get_dataset(self, id, type="train"):
        dataset = torch.load(os.path.join(self.dir, type, "data{}.pkl".format(id)))
        return dataset

    def get_data_loader(self, id, batch_size=None, type="train"):
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader

class ShiftedMNISTPartitioner():
    def __init__(self, root, save_dir):
        self.root = os.path.expanduser(root)
        self.dir = save_dir 
        # "./datasets/shifted_mnist/"
        if os.path.exists(save_dir) is not True:
            os.mkdir(save_dir)
            os.mkdir(os.path.join(save_dir, "train"))
            os.mkdir(os.path.join(save_dir, "test"))
        
    def pre_process(self, shards=100):
        # train
        mnist = datasets.MNIST(self.root, train=True, download=True)
        cid = 0
        to_tensor = transforms.ToTensor()
        for level in range(0, 10, 3):
            raw_data = []
            labels = []
            partition = random_slicing(mnist, shards)
            for x, y in mnist:
                x = to_tensor(x)
                raw_data.append(x)
                labels.append(y)

            for key, value in partition.items():
                data = [raw_data[i] for i in value]
                label = [(labels[i]+level)%10 for i in value]
                dataset = BaseDataset(data, label)
                torch.save(dataset, os.path.join(self.dir, "train" ,"data{}.pkl".format(cid)))
                cid += 1

        # test
        mnist_test = datasets.MNIST(self.root, train=False)
        for i, level in enumerate(range(0, 10, 3)):
            data = []
            label = []
            for x, y in mnist_test:
                x = to_tensor(x)
                data.append(x)
                label.append((y+level)%10)
            dataset = BaseDataset(data, label)
            torch.save(dataset, os.path.join(self.dir,"test", "data{}.pkl".format(i)))

    def get_dataset(self, id, type="train"):
        dataset = torch.load(os.path.join(self.dir, type, "data{}.pkl".format(id)))
        return dataset

    def get_data_loader(self, id, batch_size=None, type="train"):
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader


class RotatedCIFAR10Partitioner():
    def __init__(self, root, save_dir):
        self.root = os.path.expanduser(root)
        self.dir = save_dir 
        # "./datasets/rotated_mnist/"
        if os.path.exists(save_dir) is not True:
            os.mkdir(save_dir)
            os.mkdir(os.path.join(save_dir, "train"))
            os.mkdir(os.path.join(save_dir, "test"))

        self.transform  = transforms.Compose(
                                [transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
    def pre_process(self, thetas = [0, 180], shards=100):
        cifar10 = datasets.CIFAR10(self.root, train=True, download=True)
        cid = 0
        for theta in thetas:
            rotated_data = []
            partition = random_slicing(cifar10, shards)
            for x, _ in cifar10:
                x = self.transform(transforms.functional.rotate(x, theta))
                rotated_data.append(x)
            for key, value in partition.items():
                data = [rotated_data[i] for i in value]
                label = [cifar10.targets[i] for i in value]
                dataset = BaseDataset(data, label)
                torch.save(dataset, os.path.join(self.dir, "train", "data{}.pkl".format(cid)))
                cid += 1

        # test
        cifar10_test = datasets.CIFAR10(self.root, train=False)
        labels = cifar10_test.targets
        for i, theta in enumerate(thetas):
            rotated_data = []
            for x, y in cifar10_test:
                x = self.transform(transforms.functional.rotate(x, theta))
                rotated_data.append(x)
            dataset = BaseDataset(rotated_data, labels)
            torch.save(dataset, os.path.join(self.dir,"test", "data{}.pkl".format(i)))
        
    def get_dataset(self, id, type="train"):
        dataset = torch.load(os.path.join(self.dir, type, "data{}.pkl".format(id)))
        return dataset

    def get_data_loader(self, id, batch_size=None, type="train"):
        dataset = self.get_dataset(id, type)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size)
        return data_loader