import math
import numpy as np
import scipy.sparse as sp
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import random
from sklearn.model_selection import train_test_split


class S2VGraph(object):
    def __init__(self, g, label=None, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = node_features
        self.edge_mat = 0

        self.max_neighbor = 0
        

class SparseDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)


class MixedDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is None:
            if input.is_sparse:
                res = torch.sparse.mm(input, self.weight)
            else:
                res = input.matmul(self.weight)
        else:
            if input.is_sparse:
                res = torch.sparse.addmm(self.bias.expand(input.shape[0], -1), input, self.weight)
            else:
                res = torch.addmm(self.bias, input, self.weight)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None)


def sparse_matrix_to_torch(X):
    coo = X.tocoo()
    indices = np.array([coo.row, coo.col])
    return torch.sparse.FloatTensor( # torch.sparse
            torch.LongTensor(indices),
            torch.FloatTensor(coo.data),
            coo.shape)


def matrix_to_torch(X):
    if sp.issparse(X):
        return sparse_matrix_to_torch(X)
    else:
        return torch.FloatTensor(X)


def to_torch(X):
    if sp.issparse(X):
        X = to_nparray(X)
    return torch.FloatTensor(X)

def to_nparray(X):
    if sp.isspmatrix(X):
        return X.toarray()
    else: return X

def sp2adj_lists(X):
    assert sp.isspmatrix(X), 'X should be sp.sparse'
    adj_lists = []
    if sp.isspmatrix(X):
        for i in range(X.shape[0]):
            neighs = list( X[i,:].nonzero()[1] )
            adj_lists.append(neighs)
        return adj_lists
    else:
        return None


def load_dataset(dataset, data_dir='data'):
    from pathlib import Path
    import pickle
    import sys

    sys.path.append('data') # for pickle.load

    data_dir = Path(data_dir)
    suffix = '_25c.SG'
    graph_name = dataset + suffix
    path_to_file = data_dir / graph_name
    with open(path_to_file, 'rb') as f:
        graph = pickle.load(f)
    return graph


def load_latest_ckpt(model_name, dataset, ckpt_dir='./checkpoints'):
    from pathlib import Path
    ckpt_dir = Path(ckpt_dir)
    ckpt_files = []
    for p in ckpt_dir.iterdir():
        if model_name in str(p) and dataset in str(p):
            ckpt_files.append(str(p)) 
    if len(ckpt_files) > 0:
        ckpt_file = sorted(ckpt_files, key=lambda x: x[-22:])[-1] 
    else: raise FileNotFoundError
    print('checkpoint file:', ckpt_file)
    import torch
    state_dict = torch.load(ckpt_file)
    return state_dict  


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_process(adj):
    """build symmetric adjacency matrix"""
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj
    
def gin_data_preparation(dataset, num_classes=2):
    graph = load_dataset(dataset)
    
    influ_mat_list = copy.copy(graph.influ_mat_list)
    G = nx.from_scipy_sparse_matrix(graph.adj_matrix)
    degrees = np.array([val for (node, val) in G.degree()])
    
    seed_vec = influ_mat_list[:, :, 0]
    influ_vec = influ_mat_list[:, :, -1]
    
    seed_vec = [torch.FloatTensor(i) for i in seed_vec]
    influ_vec = [torch.FloatTensor(i) for i in influ_vec]
    g_list = []
    for x in seed_vec:
        temp_feature = F.one_hot(x.to(torch.long), num_classes).to(torch.float)
        g_list.append(S2VGraph(G, 0, node_tags=None, node_features=temp_feature))
    
    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        #deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
        g.edge_mat = torch.LongTensor(edges).transpose(0,1)
    
    train_g_list, test_g_list, train_x, test_x, train_y, test_y = train_test_split(g_list, seed_vec, influ_vec, test_size=0.13)
    
    return train_g_list, test_g_list, train_x, test_x, train_y, test_y


class InverseProblemDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.graph = load_dataset(dataset)
        
        self.data = self.cache(self.graph)
        
    def cache(self, graph):
        graph.influ_mat_list = graph.influ_mat_list[:50]
        influ_mat_list = copy.copy(graph.influ_mat_list)
        
        seed_vec = influ_mat_list[:, :, 0]
        influ_vec = influ_mat_list[:, :, -1]
        vec_pairs = torch.Tensor(np.stack((seed_vec, influ_vec), -1))
        return vec_pairs
        
    def __getitem__(self, item):
        vec_pair = self.data[item]
        return vec_pair
    
    def __len__(self):
        return len(self.data)

def diffusion_evaluation(adj_matrix, seed, diffusion='LT'):

    total_infect = 0
    G = nx.from_scipy_sparse_matrix(adj_matrix)
    
    for i in range(10):
        
        if diffusion == 'LT':
            model = ep.ThresholdModel(G)
            config = mc.Configuration()
            for n in G.nodes():
                config.add_node_configuration("threshold", n, 0.5)
        elif diffusion == 'IC':
            model = ep.IndependentCascadesModel(G)
            config = mc.Configuration()
            for e in G.edges():
                config.add_edge_configuration("threshold", e, 1/nx.degree(G)[e[1]])
        elif diffusion == 'SIS':
            model = ep.SISModel(G)
            config = mc.Configuration()
            config.add_model_parameter('beta', 0.001)
            config.add_model_parameter('lambda', 0.001)
        else:
            raise ValueError('Only IC, LT and SIS are supported.')

        config.add_model_initial_configuration("Infected", seed)

        model.set_initial_status(config)

        iterations = model.iteration_bunch(100)

        node_status = iterations[0]['status']

        seed_vec = np.array(list(node_status.values()))

        for j in range(1, len(iterations)):
            node_status.update(iterations[j]['status'])


        inf_vec = np.array(list(node_status.values()))
        inf_vec[inf_vec == 2] = 1

        total_infect += inf_vec.sum()
    
    return total_infect/10