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
from dgl.subgraph import DGLSubGraph
from torch.utils.data import Dataset, DataLoader
from dset import dgraph, collate
from dgl.nn.pytorch.conv import GraphConv
from torch.nn import Linear
# from experience_poor import replayer, process
# from playagent import agent, play_qlearning
# from environment import env
# from qnet import dqn
# from experience_poor import replayer
from networkx.algorithms.isomorphism import GraphMatcher, DiGraphMatcher
import networkx.algorithms.isomorphism as iso
from net import sub_GMN

device = torch.device('cuda:0')

GCN_in_size, GCN_out_size, q_size, da_size, NTN_k = 10, 128, 5, 18, 16


md = sub_GMN(GCN_in_size, GCN_out_size, q_size, da_size, NTN_k)
md.train()
md.to(device)


a = './数据/train/'
dset = dgraph(root_dir=a)

data_loader = DataLoader(dset, batch_size=2, shuffle=False, collate_fn=collate)

# nm = iso.numerical_node_match('x', 1.0)
# aaaa = torch.arange(10, dtype=torch.float32).reshape(10,1)
for i, (bbg1, bbg2, lllabel, same) in enumerate(data_loader):
    print(i)
    bbg1.to(device)
    bbg2.to(device)
    bsame = same.to(device)
    y = md(bg_da=bbg1, bg_q=bbg2, b_same=bsame)
    print(y)










    # print(bbg1)
    # print(bbg2)
    # print(lllabel)
    # print(m)
    # print(lllabel.shape)
    # print(m.shape)
