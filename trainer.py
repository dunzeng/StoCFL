from copy import deepcopy
import torch
from torch.utils.data import DataLoader

from fedlab.core.client.scale.trainer import SerialTrainer
from fedlab.utils import SerializationTool
from fedlab.utils.dataset import SubsetSampler
from fedlab.utils.functional import get_best_gpu, load_dict, evaluate

from fedscale.core.utils.femnist import FEMNIST
from fedscale.core.utils.utils_data import get_data_transform
from fedscale.core.utils.divide_data import DataPartitioner

from functional import evaluate_fairness, evaluate_personalization

class FEMNISTTrainer(SerialTrainer):
    def __init__(self, model, args, client_num=2800, cuda=True, logger=None):
        super().__init__(model, client_num, cuda, logger)

        self.args = args
        self.args.task = "cv"
        train_transform, test_transform = get_data_transform('mnist')
        train_dataset = FEMNIST('./datasets/femnist', dataset='train', transform=train_transform)
        test_dataset = FEMNIST('./datasets/femnist', dataset='test', transform=test_transform)
        self.training_sets = DataPartitioner(data=train_dataset, args=self.args, numOfClass=62)
        self.training_sets.partition_data_helper(num_clients=None, data_map_file='./datasets/femnist/client_data_mapping/train.csv')
        self.weights = self.training_sets.getSize()
        
        self.testing_sets = DataPartitioner(data=test_dataset, args=self.args, numOfClass=62, isTest=True)
        self.testing_sets.partition_data_helper(num_clients=None, data_map_file='./datasets/femnist/client_data_mapping/train.csv')

    def _get_dataloader(self, client_id):
        partition = self.training_sets.use(client_id, istest=False)
        num_loaders = min(int(len(partition)/ self.args.batch_size/2),  4)
        dataloader = DataLoader(partition, batch_size=self.args.batch_size, shuffle=True, pin_memory=True, num_workers=num_loaders, drop_last=False)
        return dataloader

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

    def local_process(self, id_list, payload):
        global_model = payload[0]
        updated_glb_models = []
        for id in id_list:
            self._LOGGER.info("Local process is running. Training client {}".format(id))
            train_loader = self._get_dataloader(id)
            glb_model  = self._train_alone(global_model, train_loader)
            updated_glb_models.append(glb_model)
        return updated_glb_models
    
class DittoTrainer(SerialTrainer):
    def __init__(self, model, dataset, data_slices, args, cuda=True, logger=None):
        super().__init__(model, args.client_num, cuda, logger)
        
        self.dataset = dataset
        self.data_slices = data_slices
        self.args = args # mu, epochs, batch_size, lr
        self.local_models = {}
        for i in range(args.client_num):
            self.local_models[i] = self.model_parameters
        self.weights = [1 for _ in range(self.client_num)]
        
    def local_process(self, id_list, payload):
        global_model = payload[0]
        updated_glb_models = []
        for id in id_list:
            self._LOGGER.info("Local process is running. Training client {}".format(id))
            train_loader = self._get_dataloader(id)
            self.local_models[id], glb_model  = self._train_alone(global_model, self.local_models[id], train_loader)
            updated_glb_models.append(glb_model)
        return updated_glb_models

    def _train_alone(self, global_model, local_model, train_loader):
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

        updated_glb_models = deepcopy(self.model_parameters)

        frz_model = deepcopy(self._model)
        SerializationTool.deserialize_model(frz_model, global_model)

        SerializationTool.deserialize_model(self._model, local_model)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=self.args.lr)

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
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self.model_parameters, updated_glb_models

    def _get_dataloader(self, client_id):
        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=SubsetSampler(indices=self.data_slices[client_id],
                                  shuffle=True),
            batch_size=self.args.batch_size)
        return train_loader

    def evaluate_personalization(self):
        stat_trloss, stat_tracc, stat_teloss, stat_teacc = [], [], [], []
        for i in range(self.client_num):
            train_loader = self._get_dataloader(i)
            SerializationTool.deserialize_model(self._model, self.local_models[i])
            train_loss, train_acc = evaluate(self._model, torch.nn.CrossEntropyLoss(), train_loader)
            stat_trloss.append(train_loss)
            stat_tracc.append(train_acc)

            test_loss, test_acc = evaluate(self._model, torch.nn.CrossEntropyLoss(), self.args.test_loader)
            stat_teloss.append(test_loss)
            stat_teacc.append(test_acc)  
        return torch.Tensor(stat_trloss), torch.Tensor(stat_tracc), torch.Tensor(stat_teloss), torch.Tensor(stat_teacc)
        
    def evaluate_fairness(self, global_model):
        SerializationTool.deserialize_model(self._model, global_model)
        stat_glb_loss, stat_glb_acc = [], []
        for i in range(self.client_num):
            train_loader = self._get_dataloader(i)
            eval_loss, eval_acc = evaluate(self._model, torch.nn.CrossEntropyLoss(), train_loader)
            stat_glb_loss.append(eval_loss), stat_glb_acc.append(eval_acc)
        return torch.Tensor(stat_glb_loss), torch.Tensor(stat_glb_acc)


class FEMNIST_DittoTrainer(FEMNISTTrainer):
    def __init__(self, model, args, client_num=2800, cuda=True, logger=None):
        super().__init__(model, args, client_num, cuda, logger)
        self.local_models = {}
        for i in range(args.client_num):
            self.local_models[i] = self.model_parameters

    def local_process(self, id_list, payload):
        global_model = payload[0]
        updated_glb_models = []
        for id in id_list:
            self._LOGGER.info("Local process is running. Training client {}".format(id))
            train_loader = self._get_dataloader(id)
            self.local_models[id], glb_model  = self._train_alone(global_model, self.local_models[id], train_loader)
            updated_glb_models.append(glb_model)
        return updated_glb_models

    def _train_alone(self, global_model, local_model, train_loader):
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

        updated_glb_models = deepcopy(self.model_parameters)

        frz_model = deepcopy(self._model)
        SerializationTool.deserialize_model(frz_model, global_model)

        SerializationTool.deserialize_model(self._model, local_model)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self._model.parameters(), lr=self.args.lr)

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
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self.model_parameters, updated_glb_models

    def evaluate_personalization(self):
        stat_trloss, stat_tracc, stat_teloss, stat_teacc = [], [], [], []
        for i in range(self.client_num):
            train_loader = self._get_dataloader(i)
            SerializationTool.deserialize_model(self._model, self.local_models[i])
            train_loss, train_acc = evaluate(self._model, torch.nn.CrossEntropyLoss(), train_loader)
            stat_trloss.append(train_loss)
            stat_tracc.append(train_acc) 

            

            test_loss, test_acc = evaluate(self._model, torch.nn.CrossEntropyLoss(), self.args.test_loader)
            stat_teloss.append(test_loss)
            stat_teacc.append(test_acc)  
        return torch.Tensor(stat_trloss), torch.Tensor(stat_tracc), torch.Tensor(stat_teloss), torch.Tensor(stat_teacc)
        
    def evaluate_fairness(self, global_model):
        SerializationTool.deserialize_model(self._model, global_model)
        stat_glb_loss, stat_glb_acc = [], []
        for i in range(self.client_num):
            train_loader = self._get_dataloader(i)
            eval_loss, eval_acc = evaluate(self._model, torch.nn.CrossEntropyLoss(), train_loader)
            stat_glb_loss.append(eval_loss), stat_glb_acc.append(eval_acc)
        return torch.Tensor(stat_glb_loss), torch.Tensor(stat_glb_acc)