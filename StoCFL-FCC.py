from copy import deepcopy
from random import random
from shutil import move
import numpy as np
from setuptools import setup
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

from models.cnn import CNN_MNIST, AlexNet_CIFAR10, CNN_FEMNIST
from models.linear import SimpleLinear
from fedlab.utils.functional import get_best_gpu, load_dict, evaluate

from settings import mnist, cifar10, femnist
from functional import *
from trainer import FEMNISTTrainer, ShiftedMNISTTrainer, RotatedMNISTTrainer, LabelSkewMNISTTrainer, FeatureSkewHybridMNISTTrainer
from datasets import BaseDataset

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
        cluster_log = []
        accuracy_log = []
        loss_log = []
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
            
            # cluster
            cluster_movements = []
            sampled_cluster, activated_bucket = self.bucket(cid_this_round, movements)
            cluster_movements = [Aggregators.fedavg_aggregate(move_list) for move_list in activated_bucket]
            self.inter_cluster_union(sampled_cluster, cluster_movements)
            self.get_cluster()

            # for clu_id, move_list in zip(sampled_cluster, activated_bucket):
            #     # cid = self.find_cluster_content[clu_id]
            #     # move_list = [self.psi[i] for i in cid]
            #     cluster_movements.append(Aggregators.fedavg_aggregate(move_list))
            
            
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
                sampled_cluster, updates_bucket = self.bucket(cid_this_round, updated_cmodels)
                for cluster, updates in zip(sampled_cluster, updates_bucket):
                    self.cluster_parameters[cluster] = Aggregators.fedavg_aggregate(updates)
                    SerializationTool.deserialize_model(self._model, self.cluster_parameters[cluster])

                    cluster_content = self.find_cluster_content[cluster]
                    test_loader = self.client_trainer.get_test_dataloader(cluster_content[0])
                    loss, acc = evaluate(self._model, torch.nn.CrossEntropyLoss(), test_loader)
                    self.args.exp_logger.info("Round {}, Cluster {} - Global Test loss: {:.4f}, acc: {:.4f}"
                                              .format(round, cluster, loss, acc))

                    # clients in cluster local test
                    loss, acc = self.client_trainer.evaluate(cluster_content, self._model, torch.nn.CrossEntropyLoss())
                    self.args.exp_logger.info("Round {}, Cluster {}, id {} - Local Test loss: {:.4f}, acc: {:.4f}"
                                              .format(round, cluster, cluster_content, loss, acc))

                    # global test
                    SerializationTool.deserialize_model(self._model, self.gmodel)
                    _, acc = evaluate(self._model, torch.nn.CrossEntropyLoss(), test_loader)
                    g_test.append(acc)

                self.args.exp_logger.info("Round {}, Global acc: {}".format(round, g_test))
                accuracy_log.append(acc)
                loss_log.append(loss)

            # record each round's result
            cluster_log.append(deepcopy(self.find_cluster_content))
            
            
        # self.save_log(cluster_log, accuracy_log, loss_log)
        # torch.save({"cluster":cluster_log, "accuracy":accuracy_log, "loss":loss_log, "unsample": unsample_log}, "{}/exp_{}_logs.pkl".format(self.args.dir, self.args.setting))
        torch.save({"cluster":cluster_log}, "{}/exp_{}_logs.pkl".format(self.args.dir, self.args.setting))



    def inter_cluster_union(self, sampled_cluster, cluster_directions):
        if len(sampled_cluster) <= 1:
            return
        # cluster_directions = [self.movements[ele] for ele in sampled_cluster] 
        M = dt_matrix(cluster_directions)

        # for id in sampled_cluster:
        #     self.args.exp_logger.info("cluster {}: clients {}".format(id, sorted(self.find_cluster_content[id])))

        sorted_key, result = parse_matrix(M, sampled_cluster, descending=True)
        
        filename = "{}/cos_rec_{}.log".format(self.args.dir, self.args.setting)
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
        filename = "{}/cluster_rec_{}.log".format(self.args.dir, self.args.setting)
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
    # python StoCFL-FCC.py --com_round 100 --num_per_round 20 --batch_size 128 --lr 0.01 --epochs 5 --mu 0.1 --tau 0.2 --setting fskewed --k 4
    parser = argparse.ArgumentParser(description="Standalone training example")
    # server
    parser.add_argument("--com_round", type=int, default=100)
    parser.add_argument("--num_per_round", type=int, default=20)

    # trainer
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--mu", type=float, default=0.1)

    # setting
    parser.add_argument("--dataset", type=float, default=0.3)
    parser.add_argument("--setting", type=str, default="shifted")  # fskewed, lskewed, shifted, rotated
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--process", type=int, default=1) # preprocess data partition and augmentation

    # cluster
    parser.add_argument("--tau", type=float, default=0.3)
    parser.add_argument("--train", type=int, default=0) # set this to 0, only clustering no learning.

    # anchor
    args = parser.parse_args()

    args.seed = 0
    setup_seed(args.seed)
    
    # feature skewed
    if args.setting == "fskewed":
        model = SimpleLinear()
        trainer = FeatureSkewHybridMNISTTrainer(deepcopy(model), args, cuda=True)
        args.k=2
        
    # label skewed
    if args.setting == "lskewed":
        model = SimpleLinear()
        trainer = LabelSkewMNISTTrainer(deepcopy(model), args, cuda=True, k=args.k)

    # feature same, label different
    if args.setting == "shifted":
        model = SimpleLinear()
        trainer = ShiftedMNISTTrainer(deepcopy(model), args, cuda=True, k=args.k)
        
    # label same, feature different
    if args.setting == "rotated":
        model = SimpleLinear()
        trainer = RotatedMNISTTrainer(deepcopy(model), args, cuda=True, k=args.k)

    if args.process:
        trainer.dataset.pre_process()

    args.time_stamp = time.strftime('%m-%d-%H:%M', time.localtime())
    dir = "./logs/runs-{}-{}-{}".format(args.setting, args.k, args.time_stamp)
    os.mkdir(dir)
    args.exp_logger = Logger("StoCFL", "{}/{}-k{}-time-{}.log".format(dir, args.setting, args.k, args.time_stamp))
    args.exp_logger.info(str(args))
    args.dir = dir
    server = StandaloneServer(deepcopy(model), True, trainer, args)
    server.main()