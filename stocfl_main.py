from copy import deepcopy
from random import random
from shutil import move
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import argparse
import random
import time

from fedlab.core.model_maintainer import ModelMaintainer
from fedlab.core.client.scale.trainer import SerialTrainer
from fedlab.utils import SerializationTool,Aggregators, Logger
from fedlab.utils.dataset import SubsetSampler

from models.cnn import CNN_MNIST, AlexNet_CIFAR10
from fedlab.utils.functional import get_best_gpu, load_dict, evaluate

from settings import mnist, cifar10, femnist

from functional import *
from trainer import FEMNIST_DittoTrainer, DittoTrainer

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
        
        self.uploaded_parameters = [torch.zeros_like(self.model_parameters) for _ in range(client_trainer.client_num)] # uploaded_parameters[id] -> model_parameters of local id

        self.cluster_parameters = [torch.zeros_like(self.model_parameters) for _ in range(client_trainer.client_num)]
        self.activated_bucket = [[] for _ in range(self.client_trainer.client_num)]
 
        self.args = args # com_round, num_per_round
        self.client_id = [i for i in range(client_trainer.client_num)]

        #self.tau_intra = self.args.t_intra
        #self.tau_inter = self.args.t_inter

        self.tau = self.args.tau

        self.cluster_logger = Logger("cluster", "./logs/cluster_info.log")
        self.weights = client_trainer.weights


    def main(self): 
        for round in range(self.args.com_round):
            global_model = SerializationTool.serialize_model(self._model)

            if len(self.client_id) < self.args.num_per_round:
                self.client_id = [i for i in range(self.client_trainer.client_num)]
            cid_this_round = sorted(random.sample(self.client_id, self.args.num_per_round))
            self.client_id = list(set(self.client_id) - set(cid_this_round))

            print("client id this round {}".format(cid_this_round))
            global_models = self.client_trainer.local_process(id_list=cid_this_round, payload=[global_model])

            for id, parameters in zip(cid_this_round, global_models):
                self.uploaded_parameters[id] = parameters

            activated_cluster = self.bucket(cid_this_round)

            for cluster_id in activated_cluster:
                client_id_list = self.activated_bucket[cluster_id]

                # debug info
                unactivated = []
                content = self.find_cluster_content[cluster_id]
                for id in content:
                    if id not in client_id_list:
                        unactivated.append(id)
                self.cluster_logger.info("chekcing cluster {}, activated clients {}, unactivate clients {}".format(cluster_id, client_id_list, unactivated))
                
                if len(client_id_list) <= 1:
                    continue

                gradients = [global_model - self.uploaded_parameters[id] for id in client_id_list]

                self.intra_re_cluster(cluster_id, client_id_list, gradients)
            
            activated_cluster = self.bucket(cid_this_round)

            for cluster in activated_cluster:
                self.calculate_cluster_parameters(cluster, self.activated_bucket[cluster])

            # re cluster
            self.inter_re_cluster(cid_this_round)
            
            # performe inter aggregation

            # cluster-wise
            if args.agg == "cavg":
                self.cluster_fedavg(activated_cluster) 

            if args.agg == "cmgda":
                self.cluster_mgda(activated_cluster)

            # alone
            if args.agg == "mgda":
                pass

            if args.agg == "avg":
                self.fedavg(cid_this_round)
            
            # self.mtl_update(activated_cluster)

            self.cluster_status()

            loss, acc = evaluate(self._model, torch.nn.CrossEntropyLoss(), self.args.test_loader)
            self.args.exp_logger.info("Round {}, Global Test loss: {:.4f}, acc: {:.4f}".format(round, loss, acc))
            
            if (round+1) % self.args.freq == 0:
                trl, tra, tel, tea = self.client_trainer.evaluate_personalization()
                self.args.exp_logger.info("Personalization Metric: Train loss: {:.4f}({:.4f}) , Train Accuracy:{:.4f}({:.4f}); Test loss: {:.4f}({:.4f}), Test Accuracy: {:.4f}({:.4f})".format(trl.mean(), trl.std(), tra.mean(), tra.std(), tel.mean(), tel.std(), tea.mean(), tea.std()))
                floss, facc = self.client_trainer.evaluate_fairness(self.model_parameters)
                self.args.exp_logger.info("Fairness Metric: Global Test loss: {:.4f}({:.4f}), Test Accuracy: {:.4f}({:.4f})".format(floss.mean(), floss.std(), facc.mean(), facc.std()))

    def intra_re_cluster(self, cluster_id, id_list, gradients):
        self.cluster_logger.info("Perform intra recluster algorithm with cluster id {} and clients {}".format(cluster_id, id_list))
        removed_mark = []
        M = dt_matrix(gradients)
        sorted_key, result = parse_matrix(M, id_list, descending=False)
        for key in sorted_key:
            cida, cidb = int(key.split(',')[0]), int(key.split(',')[1])

            if cida in removed_mark or cidb in removed_mark:
                continue

            grada = self.model_parameters - self.uploaded_parameters[cida]
            gradb = self.model_parameters - self.uploaded_parameters[cidb]

            if result[key] < self.tau:
                dt_cida = dt(grada, self.model_parameters - self.cluster_parameters[cluster_id])
                dt_cidb = dt(gradb, self.model_parameters - self.cluster_parameters[cluster_id])
                if dt_cida > dt_cidb:
                    self.depart(cidb)
                    removed_mark.append(cidb)
                else:
                    self.depart(cida)
                    removed_mark.append(cida)
            else:
                break

    def inter_re_cluster(self, client_list):
        self.cluster_logger.info("Performing inter recluster algorithm with clients {}".format(client_list))
        while True:
            activated_cluster = self.bucket(client_list)
            cluster_directions = [self.model_parameters - self.cluster_parameters[ele] for ele in activated_cluster] 
            M = dt_matrix(cluster_directions)
            
            if len(activated_cluster) <= 1:
                break

            for id in activated_cluster:
                self.cluster_logger.info("cluster {}: clients {}".format(id, self.find_cluster_content[id]))

            sorted_key, result = parse_matrix(M, activated_cluster, descending=True)
            for key in sorted_key:
                cida, cidb = int(key.split(',')[0]), int(key.split(',')[1])
                if result[key] >= self.tau:
                    self.unite(cida, cidb)
                else:
                    return

    def unite(self, i, j):
        if i > j:
            i,j = j,i

        cluster_i, cluster_j = self.find_client_cluster[i], self.find_client_cluster[j]
        if cluster_i == cluster_j:
            return
        
        self.cluster_logger.info("unite cluster {} + {}".format(cluster_i,cluster_j))
        # cluster j --> cluster i
        self.find_cluster_content[cluster_i] += self.find_cluster_content[cluster_j]
        for id in self.find_cluster_content[cluster_j]:
            self.find_client_cluster[id] = cluster_i
        self.find_cluster_content[cluster_j] = []
    
    def depart(self, i):
        cluster_id = self.find_client_cluster[i]
        self.cluster_logger.info("Removing client {} from cluster {} with content {}".format(i, cluster_id, str(self.find_cluster_content[cluster_id])))
        if cluster_id == i:
            move_list = self.find_cluster_content[cluster_id]
            move_list.remove(i)

            self.find_cluster_content[cluster_id] = [i]

            to = min(move_list)
            for cid in move_list:
                self.find_client_cluster[cid] = to
            self.find_cluster_content[to] = move_list

            self.refresh_cluster_parameters([to,i])
        else:
            self.find_client_cluster[i] = i
            self.find_cluster_content[i] = [i]
            self.find_cluster_content[cluster_id].remove(i)
            self.refresh_cluster_parameters([cluster_id,i])

    def bucket(self, client_list):
        self.activated_bucket = [[] for _ in range(self.client_trainer.client_num)]
        for id in client_list:
            self.activated_bucket[self.find_client_cluster[id]].append(id)
        activated_cluster = []
        for i, ele in enumerate(self.activated_bucket):
                if len(ele)>0:
                    activated_cluster.append(i)
        return activated_cluster

    def calculate_cluster_parameters(self, cluster, client_list=None):
        check = self.find_cluster_content[cluster] # check

        if client_list is None:
            client_list = self.find_cluster_content[cluster]
        
        if len(client_list) > 0:
            parameter_list = [self.uploaded_parameters[ele] for ele in client_list]
            sum_ = sum([self.weights[ele] for ele in client_list])
            weights = [self.weights[ele]/sum_ for ele in client_list]
            aggregated = Aggregators.fedavg_aggregate(parameter_list, weights)
            self.cluster_parameters[cluster] = aggregated


    def refresh(self, cluster_id):
        content = self.find_cluster_content[cluster_id]
        if len(content) > 0:
            parameter_list = [self.uploaded_parameters[ele] for ele in content]
            sum_ = sum([self.weights[ele] for ele in content])
            weights = [self.weights[ele]/sum_ for ele in content]
            aggregated = Aggregators.fedavg_aggregate(parameter_list, weights)
            self.cluster_parameters[cluster_id] = aggregated

    def refresh_cluster_parameters(self, cluster_id_list):
        for key in cluster_id_list:
            self.refresh(key)

    def fedavg(self, client_list):
        parameters = [self.uploaded_parameters[ele] for ele in client_list]
        weights = [self.client_trainer.weights[ele] for ele in client_list]
        aggregated = Aggregators.fedavg_aggregate(parameters, weights)
        SerializationTool.deserialize_model(self._model, aggregated)
    
    def cluster_fedavg(self, activated_cluster):
        parameters = [self.cluster_parameters[ele] for ele in activated_cluster]

        def get_weight(self, cid_list):
            weight = 0
            for id in cid_list:
                weight += self.client_trainer.weights[id]
            return weight
            
        weights = [get_weight(self, self.find_cluster_content[ele]) for ele in activated_cluster]
        aggregated = Aggregators.fedavg_aggregate(parameters, weights)
        SerializationTool.deserialize_model(self._model, aggregated)

    def cluster_mgda(self, activated_cluster):
        lambda0 = [1.0/len(activated_cluster) for _ in activated_cluster]
        directions = [self.model_parameters - self.cluster_parameters[ele] for ele in activated_cluster] 
        mgda_lambda = torch.Tensor(optim_lambdas(directions, lambda0)).view(-1)
        mgda_dt = Aggregators.fedavg_aggregate(directions, mgda_lambda)
        update_global_model = self.model_parameters - mgda_dt
        SerializationTool.deserialize_model(self._model, update_global_model)
    
    def cluster_status(self):
        result = {}
        for key, values in self.find_cluster_content.items():
            if len(values) > 0:
                self.cluster_logger.info("cluster id {} - client list {}".format(key,values))
                result.update({key: values})
                torch.save(result, "./logs/cluster_info_{}.pkl".format(self.args.time_stamp))
    

if __name__ == "__main__":
    # python dynamic-cluster.py --com_round 100 --client_num 1000 --num_per_round 50 --batch_size 64 --lr 0.1 --epochs 5 --mu 0.1 --partition noniid --dataset mnist --t_intra 0.4 --t_inter 0.9 --agg avg --freq 1000
    # python dynamic-cluster.py --com_round 100 --client_num 100 --num_per_round 10 --batch_size 128 --lr 0.1 --epochs 5 --mu 0.1 --partition noniid --dataset cifar10 --t_intra 0.4 --t_inter 0.9
    # 
    # configuration
    parser = argparse.ArgumentParser(description="Standalone training example")
    # server
    parser.add_argument("--com_round", type=int)
    parser.add_argument("--client_num", type=int)
    parser.add_argument("--num_per_round", type=int)

    # trainer
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--mu", type=float)

    # setting
    parser.add_argument("--partition", type=str)
    parser.add_argument("--dataset", type=str)

    # cluster
    parser.add_argument("--tau", type=float)

    parser.add_argument("--agg",type=str, required=True)
    parser.add_argument("--root", type=str, default="./")
    parser.add_argument("--freq", type=int, default=50)
    
    args = parser.parse_args()

    if args.dataset == "mnist":
        model, trainset, testset, data_indices, test_loader = mnist(args)
        trainer = DittoTrainer(deepcopy(model), trainset, data_indices, args)
    
    if args.dataset == "cifar10":
        model, trainset, testset, data_indices, test_loader = cifar10(args)
        trainer = DittoTrainer(deepcopy(model), trainset, data_indices, args)

    if args.dataset == "femnist":
        model, test_loader = femnist(args)
        trainer = FEMNIST_DittoTrainer(model, args, client_num=args.client_num)

    args.time_stamp = time.strftime('%m-%d-%H:%M', time.localtime())
    args.test_loader = test_loader
    args.exp_logger = Logger("DCFL", "./logs/" + "DCFL-{}-{}-time-{}.log".format(args.dataset, args.agg, args.time_stamp))
    args.exp_logger.info(str(args))
    server = StandaloneServer(deepcopy(model), True, trainer, args)
    server.main()