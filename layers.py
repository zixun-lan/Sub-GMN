import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import dgl
import networkx as nx
import dgl.function as fn
from dgl.data import MiniGCDataset
import dgl.function as fn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dgl.nn.pytorch import SumPooling
import numpy as np
from dgl.data.utils import save_graphs, get_download_dir, load_graphs
# from dgl.subgraph import DGLSubGraph
from torch.utils.data import Dataset, DataLoader
from dset import dgraph, collate
from dgl.nn.pytorch.conv import GraphConv
from torch.nn import Linear
from dgl.nn.pytorch.conv import GraphConv
from torch.nn import Linear


# class NTN(torch.nn.Module):
#     def __init__(self, gcn2out, k=16):
#         super(NTN, self).__init__()
#         self.k = k
#         self.gcn2out = gcn2out
#         self.setup_weights()
#         self.init_parameters()
#
#     def setup_weights(self):
#         """
#         Defining weights.
#         """
#         self.w = torch.nn.Parameter(torch.Tensor(self.k, self.gcn2out, self.gcn2out))
#         self.V = torch.nn.Parameter(torch.Tensor(self.k, 2 * self.gcn2out))
#         self.b = torch.nn.Parameter(torch.Tensor(self.k, 1))
#
#     def init_parameters(self):
#         """
#         Initializing weights.
#         """
#         torch.nn.init.xavier_uniform_(self.w)
#         torch.nn.init.xavier_uniform_(self.V)
#         torch.nn.init.xavier_uniform_(self.b)
#
#     def forward(self, g1_gh_em, g2_gh_em):
#         batch_size = len(g1_gh_em)
#         g2_gh_em_t = torch.transpose(g2_gh_em, 2, 3)  # part 1
#         part1 = torch.matmul(g1_gh_em, self.w)
#         part1 = torch.matmul(part1, g2_gh_em_t)
#         part1 = part1.reshape(batch_size, self.k, 1)
#
#         g1_a = g1_gh_em.reshape(batch_size, 1, self.gcn2out)
#         g2_a = g2_gh_em.reshape(batch_size, 1, self.gcn2out)
#         g1_at = torch.transpose(g1_a, 1, 2)
#         g2_at = torch.transpose(g2_a, 1, 2)
#         con = torch.cat((g1_at, g2_at), 1)
#         part2 = torch.matmul(self.V, con)
#
#         end = part1 + part2 + self.b
#         end = torch.nn.functional.sigmoid(end)
#         end = torch.transpose(end, 1, 2)
#         end = end.reshape(batch_size, self.k)
#         return end


class three_gcn(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(three_gcn, self).__init__()
        self.out_size = out_size
        self.gcn1 = GraphConv(in_feats=in_size, out_feats=self.out_size)
        self.gcn2 = GraphConv(in_feats=self.out_size, out_feats=self.out_size)
        self.gcn3 = GraphConv(in_feats=self.out_size, out_feats=self.out_size)

    def forward(self, graph, g_size):  # act nx2 的二维数组, 单张图输入一定要batch
        # print(graph)

        y = self.gcn1(graph=graph, feat=graph.ndata['x'])
        first_layer_node_em = torch.nn.functional.elu(y)
        y = self.gcn2(graph=graph, feat=first_layer_node_em)
        second_layer_node_em = torch.nn.functional.elu(y)
        y = self.gcn3(graph=graph, feat=second_layer_node_em)
        third_layer_node_em = torch.nn.functional.softmax(y, dim=1)
        # graph.ndata['x'] = third_layer_node_em
        # graph_level_em = dgl.sum_nodes(graph, 'x')  # act nx2 的二维数组, 单张图输入一定要batch
        return first_layer_node_em.reshape(-1, g_size, self.out_size), second_layer_node_em.reshape(-1, g_size, self.out_size), third_layer_node_em.reshape(-1, g_size, self.out_size)


class cross_sim(torch.nn.Module):
    def __init__(self, D):
        super(cross_sim, self).__init__()
        self.D = D
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.w = torch.nn.Parameter(torch.Tensor(self.D, self.D))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.w)

    def forward(self, batch_q_em, batch_da_em):  # batch_q_em bx5xc   batch_da_em bx18xc   torch.tensor
        T_batch_da_em = torch.transpose(batch_da_em, 1, 2)
        cross = torch.matmul(batch_q_em, self.w)
        cross = torch.matmul(cross, T_batch_da_em)
        cross = torch.sigmoid(cross).unsqueeze(1)
        return cross  # cross bx1x5x18


def att_layer(batch_q_em, batch_da_em):  # batch_q_em bx5xc   batch_da_em bx18xc   torch.tensor
    D = batch_q_em.size()[2]
    T_batch_da_em = torch.transpose(batch_da_em, 1, 2)
    att = torch.matmul(batch_q_em, T_batch_da_em)
    att = att / (D ** 0.5)
    att = torch.nn.functional.softmax(att, dim=2).unsqueeze(1)
    return att  # att bx1x5x18


class NTN(torch.nn.Module):
    def __init__(self, q_size, da_size, D, k):
        super(NTN, self).__init__()
        self.k = k
        self.D = D
        self.q_size = q_size
        self.da_size = da_size
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.w = torch.nn.Parameter(torch.Tensor(self.k, self.D, self.D))
        self.V = torch.nn.Parameter(torch.Tensor(self.k, 2 * self.D))
        self.b = torch.nn.Parameter(torch.Tensor(self.k, 1, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.w)
        torch.nn.init.xavier_uniform_(self.V)
        torch.nn.init.xavier_uniform_(self.b)

    def forward(self, batch_q_em, batch_da_em):  # batch_q_em bx5xc   batch_da_em bx18xc   torch.tensor
        batch_q_em_adddim = torch.unsqueeze(batch_q_em, 1)  # batch_q_em_adddim bx1x5xc   torch.tensor
        batch_da_em_adddim = torch.unsqueeze(batch_da_em, 1)  # batch_da_em _adddim bx1x18xc   torch.tensor
        T_batch_da_em_adddim = torch.transpose(batch_da_em_adddim, 2, 3)  # T_batch_da_em _adddim bx1xcx18   torch.tensor
        # first part
        first = torch.matmul(batch_q_em_adddim, self.w)  # first bxkx5xc   torch.tensor
        first = torch.matmul(first, T_batch_da_em_adddim)  # first bxkx5x18   torch.tensor
        # first part
        # second part
        ed_batch_q_em = torch.unsqueeze(batch_q_em, 2)  # ed_batch_q_em bx5x1xc   torch.tensor
        ed_batch_q_em = ed_batch_q_em.repeat(1, 1, self.da_size, 1)  # ed_batch_q_em bx5x18xc   torch.tensor
        ed_batch_q_em = ed_batch_q_em.reshape(-1, self.q_size * self.da_size, self.D)  # ed_batch_q_em bx90xc

        ed_batch_da_em = torch.unsqueeze(batch_da_em, 1)  # ed_batch_da_em bx1x18xc   torch.tensor
        ed_batch_da_em = ed_batch_da_em.repeat(1, self.q_size, 1, 1)  # ed_batch_da_em bx5x18xc   torch.tensor
        ed_batch_da_em = ed_batch_da_em.reshape(-1, self.q_size * self.da_size, self.D)  # ed_batch_da_em bx90xc

        mid = torch.cat([ed_batch_q_em, ed_batch_da_em], 2)  # mid bx90x2c
        mid = torch.transpose(mid, 1, 2)  # mid bx2cx90
        mid = torch.matmul(self.V, mid)  # mid bxkx90
        mid = mid.reshape(-1, self.k, self.q_size, self.da_size)  # mid bxkx5x18
        # second part
        end = first + mid + self.b
        return torch.sigmoid(end)  # end bxkx5x18
