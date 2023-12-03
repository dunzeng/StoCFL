import torch
import torchvision

from torch.utils.data import DataLoader
from torchvision import transforms

from models.cnn import CNN_MNIST, CNN_CIFAR10, CNN_FEMNIST, AlexNet_CIFAR10

from fedlab.utils.functional import load_dict

from fedscale.core.utils.femnist import FEMNIST
from fedscale.core.utils.utils_data import get_data_transform
from fedscale.core.utils.divide_data import DataPartitioner


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

    # model = AlexNet_CIFAR10()
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
    #train_dataset = FEMNIST('./datasets/femnist', dataset='train', transform=train_transform)
    test_dataset = FEMNIST('./datasets/femnist',
                           dataset='test',
                           transform=test_transform)
    #val_dataset = FEMNIST('./datasets/femnist', dataset='val', transform=test_transform)

    test_loader = DataLoader(test_dataset, batch_size=512)

    return model, test_loader
