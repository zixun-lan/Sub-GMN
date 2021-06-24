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
from sklearn.metrics import f1_score

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
batch_size = 256
epoach = 5000
device = torch.device('cuda:0')
weight_decay = 0.01
reg_ture = True

data_loader = DataLoader(dset, batch_size=batch_size, shuffle=True, collate_fn=collate)
data_test = DataLoader(d_test, batch_size=100, shuffle=False, collate_fn=collate)

model = sub_GMN(GCN_in_size, GCN_out_size, q_size, da_size, NTN_k, mask=True)
# model.load_state_dict(torch.load('./max_acc.pkl'))
model.train()
model.to(device)

reg = Regularization(model, weight_decay, p=0)
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.MSELoss().to(device)

min_test_loss = 10
max_acc = 0
f1_max = 0

for i in np.arange(epoach):
    print('epoach:  ', i)
    epoach_loss = []
    for j, (bbg1, bbg2, lllabel, same) in enumerate(data_loader):
        # print('batch:  ', j)
        bbg1 = bbg1.to(device)
        bbg2 = bbg2.to(device)
        b_lllabel = lllabel.to(device)
        b_same = same.to(device)
        y_hat = model(bg_da=bbg1, bg_q=bbg2, b_same=b_same)
        y_Hat = torch.masked_select(y_hat, torch.tensor(b_same, dtype=torch.bool).to(device))
        y_label = torch.masked_select(b_lllabel, torch.tensor(b_same, dtype=torch.bool).to(device))
        # print(y_Hat)
        # print(y_label)
        loss = criterion(y_Hat, y_label)

        # loss = criterion(y_hat, b_lllabel)
        if reg_ture:
            loss = loss + reg(model)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoach_loss.append(float(loss.detach()))
        # print(loss.detach().cpu().numpy())
        # print('batch loss:  ', float(loss.detach()))
    print('!!!!!!!!epoach_loss:  ', np.mean(epoach_loss))
    torch.save(model.state_dict(), './train.pkl')
    for k, (bbg1, bbg2, lllabel, same) in enumerate(data_test):
        print('test!!!!!!:  ', k)
        bbg1 = bbg1.to(device)
        bbg2 = bbg2.to(device)
        b_lllabel = lllabel.to(device)
        b_same = same.to(device)
        with torch.no_grad():
            y_hat = model(bg_da=bbg1, bg_q=bbg2, b_same=b_same)
            pass
        # y_hat = model(bg_da=bbg1, bg_q=bbg2, b_same=b_same)
        loss = criterion(y_hat, b_lllabel)
        print('min_test_loss:  ', min_test_loss)
        print('test_loss!!!!:  ', np.float(loss.detach()))
        pre_matching = to_predict_matching(y_hat.detach().cpu().numpy())
        print(same.numpy()[3])
        print(y_hat.detach().cpu().numpy()[3])
        print(pre_matching[3])
        acc = acc_renzao(pre_matching, q_size, da_size)
        f1 = f1_score(y_true=list(b_lllabel.cpu().numpy().reshape(-1)), y_pred=list(pre_matching.reshape(-1)),
                          average='binary', pos_label=1)
        print('max_acc      :  ', max_acc)
        print('acc          :  ', acc)
        print('f1_max       :  ', f1_max)
        print('f1           :  ', f1)

        if acc > max_acc:
            torch.save(model.state_dict(), './max_acc.pkl')
            max_acc = acc
        if np.float(loss.detach()) < min_test_loss:
            torch.save(model.state_dict(), './min_loss.pkl')
            min_test_loss = np.float(loss.detach())
        if f1 > f1_max:
            torch.save(model.state_dict(), './max_f1.pkl')
            f1_max = f1
