import sys

from fedlab.utils.functional import save_dict
from fedlab.utils.dataset.slicing import noniid_slicing, random_slicing

import torchvision

trainset = torchvision.datasets.MNIST(root="../datasets/mnist", train=True, download=False)

num_clients=1000
num_shards=1000

data_indices = noniid_slicing(trainset, num_clients=num_clients, num_shards=num_shards)
save_dict(data_indices, "mnist_noniid_{}_{}.pkl".format(num_shards, num_clients))

data_indices = random_slicing(trainset, num_clients=num_clients)
save_dict(data_indices, "mnist_iid_{}.pkl".format(num_clients))