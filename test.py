import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import dgl
import time
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
from zzh import Regularization
from utils import to_predict_matching, acc_renzao

GCN_in_size = 10
GCN_out_size = 128
q_size = 9
da_size = 18
NTN_k = 16
# a = './0.2/train/'
# b = './0.2/test/'
a = './数据/train/'
b = './数据/test/'
d_test = dgraph(root_dir=b)
dset = dgraph(root_dir=a)
batch_size = 32
epoach = 5000
device = torch.device('cuda:0')
weight_decay = 0.01
reg_ture = True

data_loader = DataLoader(dset, batch_size=batch_size, shuffle=True, collate_fn=collate)
data_test = DataLoader(d_test, batch_size=100, shuffle=False, collate_fn=collate)

model = sub_GMN(GCN_in_size, GCN_out_size, q_size, da_size, NTN_k, mask=True)
model.load_state_dict(torch.load('./mask_lzxmin_l2_5000epoach.pkl'))
model.eval()
model.to(device)

reg = Regularization(model, weight_decay, p=0)
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss().to(device)

for k, (bbg1, bbg2, lllabel, same) in enumerate(data_test):
    print('test!!!!!!:  ', k)
    bbg1.to(device)
    bbg2.to(device)
    b_lllabel = lllabel.to(device)
    b_same = same.to(device)
    t1 = time.time()
    y_hat = model(bg_da=bbg1, bg_q=bbg2, b_same=b_same)
    t2 = time.time()
    pre_matching = to_predict_matching(y_hat.detach().cpu().numpy())
    print(same.numpy()[3])
    print(y_hat.detach().cpu().numpy()[3])
    print(pre_matching[3])
    acc = acc_renzao(pre_matching, q_size, da_size)
    print(t2 - t1)
    print(acc)
    loss = criterion(y_hat, b_lllabel)
    print(loss)
