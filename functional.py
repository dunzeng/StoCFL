import numpy as np
import cvxopt
from fedlab.utils import SerializationTool
from fedlab.utils.functional import evaluate
import torch

import os
import numpy as np
import random
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms

def quadprog(Q, q, G, h, A, b):
    """
    Input: Numpy arrays, the format follows MATLAB quadprog function: https://www.mathworks.com/help/optim/ug/quadprog.html
    Output: Numpy array of the solution
    """
    Q = cvxopt.matrix(Q.tolist())
    q = cvxopt.matrix(q.tolist(), tc='d')
    G = cvxopt.matrix(G.tolist())
    h = cvxopt.matrix(h.tolist())
    A = cvxopt.matrix(A.tolist())
    b = cvxopt.matrix(b.tolist(), tc='d')
    sol = cvxopt.solvers.qp(Q, q.T, G.T, h.T, A.T, b)
    return np.array(sol['x'])

def optim_lambdas(gradients, lambda0):
    epsilon = 1
    n = len(gradients)
    J_t = [(grad/grad.norm()).numpy() for grad in gradients]
    # J_t = [grad.numpy() for grad in gradients]
    J_t = np.array(J_t)
    # target function
    Q = 2 * np.dot(J_t, J_t.T)
    q = np.array([[0] for i in range(n)])
    # equality constrint
    A = np.ones(n).T
    b = np.array([1])
    # boundary
    lb = np.array([max(0, lambda0[i] - epsilon) for i in range(n)])
    ub = np.array([min(1, lambda0[i] + epsilon) for i in range(n)])
    G = np.zeros((2 * n, n))
    for i in range(n):
        G[i][i] = -1
        G[n + i][i] = 1
    h = np.zeros((2 * n, 1))
    for i in range(n):
        h[i] = -lb[i]
        h[n + i] = ub[i]
    res = quadprog(Q, q, G, h, A, b)
    return res

def cosine_sim(grada, gradb):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    
    return cos(grada, gradb).item()

def dt(grada, gradb):
    # norm 
    assert grada.norm() > 1e-8 or gradb.norm() > 1e-8

    grada = grada/grada.norm()
    gradb = gradb/gradb.norm()

    grada = grada.numpy()
    gradb = gradb.numpy()

    if np.dot(grada, gradb) >= np.dot(gradb, gradb):
        return np.linalg.norm(gradb)
    elif np.dot(grada, gradb) >= np.dot(grada, grada):
        return np.linalg.norm(grada)
    else:
        lambda_ = np.dot((gradb - grada), gradb) / np.float_power(np.linalg.norm(gradb - grada),2)
        return np.linalg.norm(lambda_*grada + (1-lambda_)*gradb)

def dt_matrix(grad_list):
    M = [[0 for _ in range(len(grad_list))] for _ in range(len(grad_list))]
    for i, grada in enumerate(grad_list):
        for j in range(i+1, len(grad_list)):
            gradb = grad_list[j]
            # M[i][j] = dt(grada, gradb)
            M[i][j] = cosine_sim(grada, gradb)
    return M
            
def parse_matrix(M, id_list, descending=True):
    content = {}
    for i in range(len(M)):
        for j in range(i + 1, len(M)):
            content["{},{}".format(id_list[i], id_list[j])] = M[i][j]
    sorted_key = sorted(content,key=content.__getitem__, reverse=descending)
    
    # sorted_content = {}
    #for key in sorted_key:
        # sorted_content[key] = content[key]
        #print("%s:%f" % (key,content[key]))
    return sorted_key, content

def evaluate_personalization(model, parameter_list, test_loader):
    stat_loss, stat_acc = [], [], [], []
    for parameter in range(parameter_list):
        SerializationTool.deserialize_model(model, parameter)
        train_loss, train_acc = evaluate(model, torch.nn.CrossEntropyLoss(), test_loader)
        stat_loss.append(train_loss)
        stat_acc.append(train_acc)
    return torch.Tensor(stat_loss), torch.Tensor(stat_acc)
        

def evaluate_fairness(model, parameters, test_loader_list):
    SerializationTool.deserialize_model(model, parameters)
    stat_loss, stat_acc = [], []
    for test_loader in test_loader_list:
        eval_loss, eval_acc = evaluate(model, torch.nn.CrossEntropyLoss(), test_loader)
        stat_loss.append(eval_loss), stat_acc.append(eval_acc)
    return torch.Tensor(stat_loss), torch.Tensor(stat_acc)


class MnistRotated(data_utils.Dataset):
    def __init__(self, root, train=True, thetas=[0], d_label=0, download=True, transform=False):
        self.root = os.path.expanduser(root)
        self.train = train
        self.thetas = thetas
        self.d_label = d_label
        self.download = download
        self.transform = transform

        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.y_to_categorical = torch.eye(10)
        self.d_to_categorical = torch.eye(4)

        self.imgs, self.labels = self._get_data()

    def _get_data(self):
        mnist_loader = torch.utils.data.DataLoader(datasets.MNIST(self.root,
                                                                  train=self.train,
                                                                  download=self.download,
                                                                  transform=transforms.ToTensor()),
                                                   batch_size=60000,
                                                   shuffle=False)

        for i, (x, y) in enumerate(mnist_loader):
            mnist_imgs = x
            mnist_labels = y

        pil_list = []
        for x in mnist_imgs:
            pil_list.append(self.to_pil(x))

        return pil_list, mnist_labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.imgs[index]
        y = self.labels[index]

        d = np.random.choice(range(len(self.thetas)))

        if self.transform: # data augmentation random rotation by +- 90 degrees
            pass
            # random_rotation = np.random.randint(0, 360, 1)
            # return self.to_tensor(transforms.functional.rotate(x, self.thetas[d] + random_rotation)), self.y_to_categorical[y], \
            #        self.d_to_categorical[self.d_label]
        else:
            return self.to_tensor(transforms.functional.rotate(x, self.thetas[d])), self.y_to_categorical[y], self.d_to_categorical[self.d_label]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True