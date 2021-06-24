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
from networkx.algorithms.isomorphism import GraphMatcher, DiGraphMatcher
import networkx.algorithms.isomorphism as iso


def shuju(num, n_da, n_xiao, P=0.4, sloop = False):
    g1 = []
    g2 = []
    nm = iso.numerical_node_match('x', 1.0)
    while len(g1) < num:
        sub = list(np.arange(n_xiao))
        gg1 = nx.generators.random_graphs.fast_gnp_random_graph(n=n_da, p=P)
        sub_g = nx.induced_subgraph(gg1, sub)
        if nx.is_connected(gg1) is True and nx.is_connected(sub_g) is True:
            ggg1 = dgl.DGLGraph()
            ggg1.from_networkx(gg1)
            ggg1.ndata['x'] = torch.randint(1, 11, (n_da, 1), dtype=torch.float32)
            if sloop:
                ggg1.add_edges(ggg1.nodes(), ggg1.nodes())
            ggg2 = ggg1.subgraph(sub)
            ggg2.copy_from_parent()
            nx1 = ggg1.to_networkx(node_attrs=['x'])
            nx2 = ggg2.to_networkx(node_attrs=['x'])
            mm = DiGraphMatcher(G1=nx1, G2=nx2, node_match=nm)
            aa = list(mm.subgraph_isomorphisms_iter())
            if len(aa) == 1:
                g1.append(ggg1)
                g2.append(ggg2)
    a = torch.zeros(n_xiao, 1, dtype=torch.long)
    b = torch.ones(n_da - n_xiao, 1, dtype=torch.long)
    c = torch.cat((a, b), 0)
    label = c.repeat(num, 1, 1)
    return g1, g2, label


g1, g2, label = shuju(num=200, n_da=18, n_xiao=7, sloop=False)

graph_labels = {"glabel": label}
save_graphs("./数据/0200te_18_7_g1.bin", g1, graph_labels)
save_graphs("./数据/0200te_18_7_g2.bin", g2)



