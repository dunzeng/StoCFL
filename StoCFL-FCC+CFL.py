from copy import deepcopy
from random import random
from shutil import move
import numpy as np
from setuptools import setup
from sklearn.metrics import accuracy_score
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import random
import cvxopt
import time
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from fedlab.core.model_maintainer import ModelMaintainer
from fedlab.core.client.serial_trainer import SerialTrainer
from fedlab.utils import SerializationTool,Aggregators, Logger
from fedlab.utils.dataset import SubsetSampler

from models.cnn import CNN_CIFAR10, CNN_MNIST, AlexNet_CIFAR10, CNN_FEMNIST
from models.linear import SimpleLinear
from fedlab.utils.functional import get_best_gpu, load_dict, evaluate

from functional import *
from trainer import FEMNISTTrainer, ShiftedMNISTTrainer, RotatedMNISTTrainer, LabelSkewMNISTTrainer, FeatureSkewHybridMNISTTrainer
from datasets import BaseDataset, RotatedMNIST, RotatedCIFAR10Partitioner

class StandaloneServer(ModelMaintainer):
    def __init__(self, model, cuda, client_trainer, args) -> None:
        super().__init__(model, cuda)
        self.client_trainer = client_trainer

        self.client_num = self.client_trainer.client_num
        self.find_cluster_content = {} # find_cluster_content[cluster_id] -> [id_1, id_2, ...]
        self.find_client_cluster = [i for i in range(client_trainer.client_num)] # find_client_cluster[cid] -> cluster id

        for i in range(client_trainer.client_num):
            self.find_cluster_content[i] = []
            self.find_cluster_content[i].append(i)
        
        self.cluster_parameters = {}
        self.psi = {}

        self.gmodel = deepcopy(self.model_parameters)

        self.args = args # com_round, num_per_round
        self.client_id = [i for i in range(client_trainer.client_num)]

        self.tau = args.tau
        self.anchor = SerializationTool.serialize_model(self._model)
        
    def main(self): 
        cluster_log = [] # [cluster info at round i]
        accuracy_log = [] # [{"cluster":[],"global":[]}
        for round in range(self.args.com_round):
            cid_this_round = sorted(random.sample(self.client_id, self.args.num_per_round))
            self.args.exp_logger.info("Starting round {}/{}, client id this round {}".format(round, self.args.com_round, cid_this_round))

            # anchor move
            unobserved = []
            for cid in cid_this_round:
                if cid not in self.psi.keys():
                    unobserved.append(cid)
            movements = self.client_trainer.observe(unobserved, self.anchor)
            for cid, move in zip(unobserved, movements):
                self.psi[cid] = move
            movements = [self.psi[i] for i in cid_this_round]
            torch.save(self.psi, "{}/psi.pkl".format(self.args.dir))

            # cluster
            cluster_movements = []
            sampled_cluster, activated_bucket = self.bucket(cid_this_round, movements)

            # reupdate center
            # for clu_id, move_list in zip(sampled_cluster, activated_bucket):
            #     cid = self.find_cluster_content[clu_id]
            #     move_list = [self.psi[i] for i in cid]
            #     cluster_movements.append(Aggregators.fedavg_aggregate(move_list))

            cluster_movements = [Aggregators.fedavg_aggregate(move_list) for move_list in activated_bucket]
            self.inter_cluster_union(sampled_cluster, cluster_movements)
            self.get_cluster()
            cluster_log.append(deepcopy(self.find_cluster_content))
            torch.save(cluster_log, "{}/cluster.pkl".format(self.args.dir))

            # clustered federated optimization
            if self.args.train:
                for id in cid_this_round:
                    if self.find_client_cluster[id] not in self.cluster_parameters.keys():
                        self.cluster_parameters[self.find_client_cluster[id]] = deepcopy(self.gmodel)
                cluster_models = [self.cluster_parameters[self.find_client_cluster[i]] for i in cid_this_round]
                updated_cmodels, gmodels = self.client_trainer.train(cid_this_round, cluster_models, self.gmodel)

                # update global models
                self.gmodel = Aggregators.fedavg_aggregate(gmodels)

                # update cluster models
                g_test = []
                accuracy = {"cluster":[], "global": None}
                sampled_cluster, updates_bucket = self.bucket(cid_this_round, updated_cmodels)
                for cluster, updates in zip(sampled_cluster, updates_bucket):
                    self.cluster_parameters[cluster] = Aggregators.fedavg_aggregate(updates)
                    SerializationTool.deserialize_model(self._model, self.cluster_parameters[cluster])

                    cluster_content = self.find_cluster_content[cluster]
                    test_loader = self.client_trainer.get_test_dataloader(int(cluster_content[0]*self.args.k/self.args.n))
                    loss, acc = evaluate(self._model, torch.nn.CrossEntropyLoss(), test_loader)
                    self.args.exp_logger.info("Round {}, Cluster {} - Global Test loss: {:.4f}, acc: {:.4f}".format(round, cluster, loss, acc))
                    accuracy["cluster"].append(acc)

                    # global test
                    SerializationTool.deserialize_model(self._model, self.gmodel)
                    _, acc = evaluate(self._model, torch.nn.CrossEntropyLoss(), test_loader)
                    g_test.append(acc)
                accuracy["global"] = g_test
                accuracy_log.append(accuracy)
                torch.save(accuracy_log, "{}/accuracy.pkl".format(self.args.dir))
                self.args.exp_logger.info("Round {}, Global acc: {}".format(round, g_test))

    def inter_cluster_union(self, sampled_cluster, cluster_directions):
        if len(sampled_cluster) <= 1:
            return
        M = dt_matrix(cluster_directions)
        sorted_key, result = parse_matrix(M, sampled_cluster, descending=True)
        
        filename = "{}/cos_rec.log".format(self.args.dir)
        if os.path.exists(filename):
            os.remove(filename)
        matrix_logger = Logger("matrix",filename)
        # load log
        for key in sorted_key:
            # sorted_content[key] = content[key]
            matrix_logger.info("%s:%f" % (key,result[key]))

        for key in sorted_key:
            cida, cidb = int(key.split(',')[0]), int(key.split(',')[1]) # cluster id
            if result[key] >= self.tau:
                self.unite(cida, cidb)
            else:
                break

    # cluster i, cluster j
    def unite(self, i, j):
        if i not in self.find_cluster_content[i]:
            i = self.find_client_cluster[i]
        
        if j not in self.find_cluster_content[j]:
            j = self.find_client_cluster[j]

        if i==j:
            return
        
        if i > j:
            i,j = j,i

        # merge cluster model
        if len(self.find_cluster_content[i]) < len(self.find_cluster_content[j]):
            if j in self.cluster_parameters.keys():
                self.cluster_parameters[i] = self.cluster_parameters[j]

        self.find_cluster_content[i] += self.find_cluster_content[j]
        for id in self.find_cluster_content[j]:
            self.find_client_cluster[id] = i
        self.find_cluster_content[j] = [] # empty

        if j in self.cluster_parameters.keys():
            self.cluster_parameters.pop(j)

    def bucket(self, client_list, movements):
        activated_bucket = [[] for _ in range(self.client_trainer.client_num)]
        for id, move in zip(client_list, movements):
            activated_bucket[self.find_client_cluster[id]].append(move)

        sampled_cluster = []
        bucket = []
        for i, ele in enumerate(activated_bucket):
                if len(ele)>0:
                    sampled_cluster.append(i)
                    bucket.append(deepcopy(ele))
        return sampled_cluster, bucket

    def get_cluster(self):
        filename = "{}/cluster_rec.log".format(self.args.dir)
        if os.path.exists(filename):
            os.remove(filename)
        cluster_logger = Logger("cluster",filename)

        check=0
        for key, values in self.find_cluster_content.items():
            if len(values) > 0:
                cluster_logger.info("cluster id {} - {}".format(key, sorted(values)))
                check+=len(values)
        assert check == self.client_num

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone training example")
    # server
    parser.add_argument("--n", type=int) # the number of clients
    parser.add_argument("--com_round", type=int)
    parser.add_argument("--num_per_round", type=int)

    # trainer
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--mu", type=float, default=0.05)

    # cluster
    parser.add_argument("--obep", type=int, default=10)
    parser.add_argument("--obbs", type=int, default=None)

    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--train", type=int, default=1) # set this to 0, only clustering no learning.
    parser.add_argument("--seed", type=int)

    parser.add_argument("--dataset", type=str, default="cifar") # mnist, cifar
    # datset
    parser.add_argument("--process", type=int, default=0)

    # anchor
    args = parser.parse_args()

    #args.seed = 0 # [0, 42, 1998, 4421380, 789190]
    setup_seed(args.seed)
    
    args.root = "./datasets/{}/".format(args.dataset)
    args.save_dir = "./datasets/rotated_{}_{}_{}/".format(args.dataset, args.n, args.seed)

    if args.process:
        print("Preprocessing datasets...")
        if args.dataset == "mnist":
            dataset = RotatedMNIST(args.root, args.save_dir)
            # dataset = RotatedMNISTPartitioner(args.root, args.save_dir)
            dataset.pre_process(shards=int(args.n/4))
        if args.dataset == "cifar":
            dataset = RotatedCIFAR10Partitioner(args.root, args.save_dir)
            dataset.pre_process(shards=int(args.n/2))
    
    if args.dataset == "mnist":
        args.k = 4
        model = SimpleLinear()
        trainer = RotatedMNISTTrainer(deepcopy(model), args, cuda=True)

    if args.dataset == "cifar":
        args.k = 2
        model = CNN_CIFAR10()
        trainer = RotatedMNISTTrainer(deepcopy(model), args, cuda=True)
    
    args.time_stamp = time.strftime('%m-%d-%H:%M', time.localtime())
    dir = "./logs/runs-rotated-{}-n{}-p{}-seed{}-lambda{}-time-{}".format(args.dataset, args.n, args.num_per_round,args.seed, args.mu, args.time_stamp)
    os.mkdir(dir)
    args.exp_logger = Logger("StoCFL", "{}/rotated_{}.log".format(dir, args.dataset))
    args.exp_logger.info(str(args))
    args.dir = dir
    server = StandaloneServer(deepcopy(model), True, trainer, args)
    server.main()