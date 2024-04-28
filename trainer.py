from copy import deepcopy
from tqdm import tqdm
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from fedlab.core.client.serial_trainer import SerialTrainer
from fedlab.utils import SerializationTool
from fedlab.utils.dataset import SubsetSampler
from fedlab.utils.functional import get_best_gpu, load_dict, evaluate, AverageMeter
from fedlab.utils.dataset.slicing import random_slicing

from functional import evaluate_fairness, evaluate_personalization
from settings import ShiftedMNIST, RotatedMNIST, label_skew_parition, FskewedFashionMNIST
from leaf.dataloader import get_LEAF_dataloader, get_LEAF_all_test_dataloader
from leaf.pickle_dataset import PickleDataset
from datasets import RotatedCIFAR10Partitioner, RotatedMNIST, BaseDataset, ShiftedMNISTPartitioner, HybridMNISTPartitioner

class FEMNISTTrainer(SerialTrainer):
    def __init__(self, model, args, cuda=True, logger=None):
        super().__init__(model, None, cuda, logger)

        self.args = args
        self.client_num = args.client_num
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr=self.args.lr)
        self.femnist = PickleDataset(dataset_name="femnist", data_root="./datasets/femnist", pickle_root="./leaf/pickle_datasets")

    def _get_dataloader(self, client_id, batch_size):
        trainset = self.femnist.get_dataset_pickle("train", client_id)
        trainloader = trainloader = torch.utils.data.DataLoader(
                                        trainset,
                                        batch_size=batch_size,
                                        drop_last=False)
        return trainloader
        
    def observe(self, id_list, anchor):
        movements = []
        for id in id_list:
            train_loader = self._get_dataloader(id, self.args.obbs)
            move = self.ob_train(anchor, train_loader)
            movements.append(move)
        return movements

    def ob_train(self, model_parameters, train_loader):
        criterion = torch.nn.CrossEntropyLoss()
        SerializationTool.deserialize_model(self._model, model_parameters)
        self._model.train()
        for ep in range(self.args.obep):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.gpu), label.cuda(self.gpu)

                preds = self._model(data)
                loss = criterion(preds,label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return model_parameters - self.model_parameters

    def train(self, id_list, cluster_models, gmodel):
        updated_cmodels = []
        for id, cmodel in zip(tqdm(id_list), cluster_models):
            train_loader = self._get_dataloader(id, self.args.batch_size)    
            cmodel  = self.train_alone_prox(cmodel, gmodel, train_loader)
            updated_cmodels.append(cmodel)

        gmodels = []
        for id in tqdm(id_list):
            train_loader = self._get_dataloader(id, self.args.batch_size)
            gmodel = self._train_alone(gmodel, train_loader)
            gmodels.append(gmodel)

        return updated_cmodels, gmodels

    def _train_alone(self, global_model, train_loader):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=self.args.lr)
        SerializationTool.deserialize_model(self._model, global_model)
        self._model.train()
        for ep in range(self.args.epochs):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.gpu), label.cuda(self.gpu)

                preds = self._model(data)
                loss = criterion(preds,label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self.model_parameters

    def train_alone_prox(self, model_parameters, gmodel, train_loader):
        criterion = torch.nn.CrossEntropyLoss()
        SerializationTool.deserialize_model(self._model, model_parameters)

        frz_model = deepcopy(self._model)
        SerializationTool.deserialize_model(frz_model, gmodel)
        
        self._model.train()
        for ep in range(self.args.epochs):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.gpu), label.cuda(self.gpu)

                preds = self._model(data)
                l1 = criterion(preds,label)
                l2 = 0.0
                for w0, w in zip(frz_model.parameters(), self._model.parameters()):
                    l2 += torch.sum(torch.pow(w - w0, 2))

                loss = l1 + 0.5 * self.args.mu * l2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return self.model_parameters

class ShiftedMNISTTrainer(SerialTrainer):
    def __init__(self, model, args, cuda=False, logger=None, k=4):
        super().__init__(model, None, cuda, logger)

        self.args = args
        self.dataset = ShiftedMNISTPartitioner(root="./datasets/mnist/", save_dir="./datasets/shifted_mnist/")
        self.client_num = 400
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr=self.args.lr)

    def _get_dataloader(self, client_id, train_frac=0.8, batch_size=None):
        dataset = self.dataset.get_dataset(client_id)
        n_train = int(len(dataset) * train_frac)
        n_eval = len(dataset) - n_train
        data_train, data_eval = torch.utils.data.random_split(dataset, [n_train, n_eval])
        batch_size = len(dataset) if batch_size is None else batch_size
        train_loader = DataLoader(data_train, batch_size=batch_size)
        eval_loader = DataLoader(data_eval, batch_size=batch_size)

        return train_loader, eval_loader

        # data_loader = self.dataset.get_data_loader(client_id)
        # return data_loader
    
    def observe(self, id_list, anchor):
        movements = []
        for id in id_list:
            train_loader, _ = self._get_dataloader(id)
            move = self.ob_train(anchor, train_loader)
            movements.append(move)
        return movements

    def ob_train(self, model_parameters, train_loader):
        criterion = torch.nn.CrossEntropyLoss()
        SerializationTool.deserialize_model(self._model, model_parameters)
        self._model.train()
        for ep in range(1):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.gpu), label.cuda(self.gpu)

                preds = self._model(data)
                loss = criterion(preds,label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return model_parameters - self.model_parameters

    def train(self, id_list, cluster_models, gmodel):
        updated_cmodels = []
        # update each cluster model with the consideration of global model
        for id, cmodel in zip(tqdm(id_list), cluster_models):
            train_loader, _ = self._get_dataloader(id)
            cmodel  = self.train_alone_prox(cmodel, gmodel, train_loader)
            updated_cmodels.append(cmodel)

        gmodels = []
        for id in tqdm(id_list):
            train_loader, _ = self._get_dataloader(id)
            gmodel = self._train_alone(gmodel, train_loader)
            gmodels.append(gmodel)

        return updated_cmodels, gmodels

    def get_test_dataloader(self, client_id):
        id = int(client_id/5)  # 20clients
        data_loader = self.dataset.get_data_loader(id, batch_size=4096, type="test")
        return data_loader

    def _train_alone(self, model_parameters, train_loader):
        criterion = torch.nn.CrossEntropyLoss()
        SerializationTool.deserialize_model(self._model, model_parameters)
        self._model.train()
        for ep in range(self.args.epochs):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.gpu), label.cuda(self.gpu)
                preds = self._model(data)
                loss = criterion(preds,label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.model_parameters

    def train_alone_prox(self, model_parameters, gmodel, train_loader):
        criterion = torch.nn.CrossEntropyLoss()
        SerializationTool.deserialize_model(self._model, model_parameters)

        frz_model = deepcopy(self._model)
        SerializationTool.deserialize_model(frz_model, gmodel)
        
        self._model.train()
        for ep in range(self.args.epochs):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.gpu), label.cuda(self.gpu)

                preds = self._model(data)
                l1 = criterion(preds,label)
                l2 = 0.0
                for w0, w in zip(frz_model.parameters(), self._model.parameters()):
                    l2 += torch.sum(torch.pow(w - w0, 2))

                loss = l1 + 0.5 * self.args.mu * l2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return self.model_parameters

    def evaluate(self, id_list, model, criterion):
        model.eval()
        gpu = next(model.parameters()).device

        loss_ = AverageMeter()
        acc_ = AverageMeter()
        with torch.no_grad():
            for id in range(id_list):
                _, eval_loader = self._get_dataloader(id)
                for inputs, labels in eval_loader:
                    inputs = inputs.to(gpu)
                    labels = labels.to(gpu)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs, 1)
                    loss_.update(loss.item())
                    acc_.update(torch.sum(predicted.eq(labels)).item(), len(labels))

        return loss_.sum, acc_.avg

class RotatedMNISTTrainer(SerialTrainer):
    def __init__(self, model, args, cuda=False, logger=None, k=4):
        super().__init__(model, 0, cuda, logger)
        self.args = args
        if args.dataset == "mnist": 
            self.dataset = RotatedMNIST(root=args.root, save_dir=args.save_dir)
        if args.dataset == "cifar":
            self.dataset = RotatedCIFAR10Partitioner(root=args.root, save_dir=args.save_dir)
        
        self.client_num = args.n
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr=self.args.lr)

    def train(self, id_list, cluster_models, gmodel):
        updated_cmodels = []
        for id, cmodel in zip(tqdm(id_list), cluster_models):
            train_loader = self.dataset.get_data_loader(id, self.args.batch_size)    
            cmodel  = self.train_alone_prox(cmodel, gmodel, train_loader)
            updated_cmodels.append(cmodel)

        gmodels = []
        for id in tqdm(id_list):
            train_loader = self.dataset.get_data_loader(id, self.args.batch_size)  
            gmodel = self._train_alone(gmodel, train_loader)
            gmodels.append(gmodel)

        return updated_cmodels, gmodels

    def observe(self, id_list, anchor):
        movements = []
        for id in id_list:
            train_loader = self.dataset.get_data_loader(id, self.args.obbs)
            move = self.ob_train(anchor, train_loader)
            movements.append(move)
        return movements

    def ob_train(self, model_parameters, train_loader):
        criterion = torch.nn.CrossEntropyLoss()
        SerializationTool.deserialize_model(self._model, model_parameters)
        self._model.train()
        for ep in range(self.args.obep):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.gpu), label.cuda(self.gpu)
                preds = self._model(data)
                loss = criterion(preds,label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return model_parameters - self.model_parameters

    def get_test_dataloader(self, cid):
        data_loader = self.dataset.get_data_loader(cid, batch_size=4096, type="test")
        return data_loader

    def _train_alone(self, model_parameters, train_loader):
        criterion = torch.nn.CrossEntropyLoss()
        SerializationTool.deserialize_model(self._model, model_parameters)
        self._model.train()
        for ep in range(self.args.epochs):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.gpu), label.cuda(self.gpu)
                preds = self._model(data)
                loss = criterion(preds,label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.model_parameters

    def train_alone_prox(self, model_parameters, gmodel, train_loader):
        criterion = torch.nn.CrossEntropyLoss()
        SerializationTool.deserialize_model(self._model, model_parameters)

        frz_model = deepcopy(self._model)
        SerializationTool.deserialize_model(frz_model, gmodel)
        
        self._model.train()
        for ep in range(self.args.epochs):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.gpu), label.cuda(self.gpu)

                preds = self._model(data)
                l1 = criterion(preds,label)
                l2 = 0.0
                for w0, w in zip(frz_model.parameters(), self._model.parameters()):
                    l2 += torch.sum(torch.pow(w - w0, 2))

                loss = l1 + 0.5 * self.args.mu * l2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.model_parameters

class LabelSkewMNISTTrainer(SerialTrainer):
    def __init__(self, model, args, cuda=False, logger=None, k=4):
        super().__init__(model, 0, cuda, logger)
        
        self.args = args
        self.mnist = torchvision.datasets.MNIST("./dataset/mnist/", train=True, transform=transforms.ToTensor())
        self.test_set = torchvision.datasets.MNIST("./dataset/mnist/", train=False, transform=transforms.ToTensor())

        self.data_indices = label_skew_parition(self.mnist, k)
        #indices = [len(v) for k,v in self.data_indices.items()]
        self.client_num = len(self.data_indices)
        #print(indices, self.client_num)
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr=self.args.lr)

    def _get_dataloader(self, client_id):
        data_loader = DataLoader(self.mnist, batch_size=self.args.batch_size, sampler=SubsetSampler(self.data_indices[client_id]), num_workers=2)
        return data_loader

    def observe(self, id_list, anchor):
        movements = []
        for id in id_list:
            train_loader = self._get_dataloader(id)
            move = self.ob_train(anchor, train_loader)
            movements.append(move)
        return movements

    def ob_train(self, model_parameters, train_loader):
        criterion = torch.nn.CrossEntropyLoss()
        SerializationTool.deserialize_model(self._model, model_parameters)
        self._model.train()
        for ep in range(1):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.gpu), label.cuda(self.gpu)
                preds = self._model(data)
                loss = criterion(preds,label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return model_parameters - self.model_parameters

    def train(self, id_list, cluster_models, gmodel):
        updated_cmodels = []
        for id, cmodel in zip(tqdm(id_list), cluster_models):
            train_loader = self._get_dataloader(id)    
            cmodel  = self.train_alone_prox(cmodel, gmodel, train_loader)
            updated_cmodels.append(cmodel)

        gmodels = []
        for id in tqdm(id_list):
            train_loader = self._get_dataloader(id)
            gmodel = self._train_alone(gmodel, train_loader)
            gmodels.append(gmodel)

        return updated_cmodels, gmodels

    def local_process(self, id_list, payload):
        cluster_models = payload[0]
        anchor = payload[1]
        updated_models = []
        movements = []
        for id, cmodel in zip(id_list, cluster_models):
            self._LOGGER.info("Local process is running. Training client {}".format(id))
            train_loader = self._get_dataloader(id)
            
            glb_model  = self._train_alone(cmodel, train_loader)
            updated_models.append(glb_model)

            move = self._train_alone(anchor, train_loader)
            move = anchor - move
            movements.append(move)
        return updated_models, movements

    def train_alone_prox(self, model_parameters, gmodel, train_loader):
        criterion = torch.nn.CrossEntropyLoss()
        SerializationTool.deserialize_model(self._model, model_parameters)

        frz_model = deepcopy(self._model)
        SerializationTool.deserialize_model(frz_model, gmodel)
        
        self._model.train()
        for ep in range(self.args.epochs):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.gpu), label.cuda(self.gpu)

                preds = self._model(data)
                l1 = criterion(preds,label)
                l2 = 0.0
                for w0, w in zip(frz_model.parameters(), self._model.parameters()):
                    l2 += torch.sum(torch.pow(w - w0, 2))

                loss = l1 + 0.5 * self.args.mu * l2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return self.model_parameters

    def _train_alone(self, model_parameters, train_loader):
        criterion = torch.nn.CrossEntropyLoss()
        SerializationTool.deserialize_model(self._model, model_parameters)
        self._model.train()
        for ep in range(self.args.epochs):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.gpu), label.cuda(self.gpu)
                preds = self._model(data)
                loss = criterion(preds,label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.model_parameters

class FeatureSkewHybridMNISTTrainer(SerialTrainer):
    def __init__(self, model, args, cuda=False, logger=None, k=4):
        super().__init__(model, 0, cuda, logger)
        
        self.args = args

        self.dataset = HybridMNISTPartitioner(root="./datasets/mnist/", save_dir="./datasets/hybrid_mnist/")
        self.client_num = 200
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr=self.args.lr)

    def observe(self, id_list, anchor):
        movements = []
        for id in id_list:
            train_loader = self.dataset.get_data_loader(id)
            move = self.ob_train(anchor, train_loader)
            movements.append(move)
        return movements

    def ob_train(self, model_parameters, train_loader):
        criterion = torch.nn.CrossEntropyLoss()
        SerializationTool.deserialize_model(self._model, model_parameters)
        self._model.train()
        for ep in range(5):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.gpu), label.cuda(self.gpu)

                preds = self._model(data)
                loss = criterion(preds,label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return model_parameters - self.model_parameters

    def get_test_dataloader(self, cid):
        data_loader = self.dataset.get_data_loader(cid, batch_size=4096, type="test")
        return data_loader

    def train(self, id_list, cluster_models, gmodel):
        updated_cmodels = []
        for id, cmodel in zip(tqdm(id_list), cluster_models):
            train_loader = self.dataset.get_data_loader(id, self.args.batch_size)    
            cmodel  = self.train_alone_prox(cmodel, gmodel, train_loader)
            updated_cmodels.append(cmodel)

        gmodels = []
        for id in tqdm(id_list):
            train_loader = self.dataset.get_data_loader(id, self.args.batch_size)
            gmodel = self._train_alone(gmodel, train_loader)
            gmodels.append(gmodel)

        return updated_cmodels, gmodels

    def _train_alone(self, global_model, train_loader):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=self.args.lr)
        SerializationTool.deserialize_model(self._model, global_model)
        self._model.train()
        for ep in range(self.args.epochs):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.gpu), label.cuda(self.gpu)

                preds = self._model(data)
                loss = criterion(preds,label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self.model_parameters

    def train_alone_prox(self, model_parameters, gmodel, train_loader):
        criterion = torch.nn.CrossEntropyLoss()
        SerializationTool.deserialize_model(self._model, model_parameters)

        frz_model = deepcopy(self._model)
        SerializationTool.deserialize_model(frz_model, gmodel)
        
        self._model.train()
        for ep in range(self.args.epochs):
            for data, label in train_loader:
                if self.cuda:
                    data, label = data.cuda(self.gpu), label.cuda(self.gpu)

                preds = self._model(data)
                l1 = criterion(preds,label)
                l2 = 0.0
                for w0, w in zip(frz_model.parameters(), self._model.parameters()):
                    l2 += torch.sum(torch.pow(w - w0, 2))

                loss = l1 + 0.5 * self.args.mu * l2
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return self.model_parameters
