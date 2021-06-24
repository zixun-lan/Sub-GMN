import torch
import dgl
import networkx as nx
import numpy as np
from dgl.data.utils import save_graphs
from 数据生成 import *


def datadata(core_size, whole_size, n, ppath, self_loop=False, p=0.3, p_q=1):
    nm = iso.numerical_node_match('x', 1.0)
    E = np.eye(10)  # E是10x10的单位矩阵 表示不同特征对应的独热编码
    # F = np.random.randint(0, 10, size=whole_size)
    # da_feature = torch.tensor(E[F], dtype=torch.float32)
    # q_feature = da_feature[0:n]
    i = 0
    while i < n:
        print(i)
        F = np.random.randint(0, 10, size=whole_size)
        da_feature = torch.tensor(E[F], dtype=torch.float32)
        q_feature = da_feature[0:core_size]
        cccc = False

        while cccc is not True:
            g_da = nx.generators.random_graphs.fast_gnp_random_graph(n=whole_size, p=p)
            aaaa = nx.is_connected(g_da)
            g_q = nx.subgraph(g_da, list(np.arange(core_size)))
            bbbb = nx.is_connected(g_q)
            if aaaa is True:
                if bbbb is True:
                    cccc = True

        if self_loop:
            e = np.eye(whole_size)
            adj_da = nx.to_numpy_matrix(g_da) + e
        else:
            adj_da = nx.to_numpy_matrix(g_da)

        adj_q = adj_da[0:core_size, 0:core_size]

        G_da = nx.from_numpy_matrix(adj_da)
        G_q = nx.from_numpy_matrix(adj_q)

        # D_da = dgl.DGLGraph()
        # D_q = dgl.DGLGraph()

        D_da = dgl.DGLGraph(G_da)
        D_q = dgl.DGLGraph(G_q)

        D_da.ndata['x'] = da_feature
        D_q.ndata['x'] = q_feature

        test_da = dgl.DGLGraph(G_da)
        test_q = dgl.DGLGraph(G_q)
        test_da.ndata['x'] = torch.tensor(F, dtype=torch.float32).reshape(-1, 1)
        test_q.ndata['x'] = torch.tensor(F[0:core_size], dtype=torch.float32).reshape(-1, 1)

        # 检查是否唯一子图
        nx_da = test_da.to_networkx(node_attrs=['x'])
        nx_q = test_q.to_networkx(node_attrs=['x'])
        mm = DiGraphMatcher(G1=nx_da, G2=nx_q, node_match=nm)
        aa = list(mm.subgraph_isomorphisms_iter())
        print(aa)
        if len(aa) == 1:
            path = ppath + str(i) + '.bin'

            same_m = same(D_q.ndata['x'].numpy(), D_da.ndata['x'].numpy())
            m = to_m(label=np.arange(core_size), q_size=core_size, g_size=whole_size)

            graph_labels = {'glabel': torch.tensor(m, dtype=torch.float32),
                            'same_m': torch.tensor(same_m, dtype=torch.float32)}

            D_da.ndata['x'] = torch.tensor(D_da.ndata['x'], dtype=torch.float32)
            D_q.ndata['x'] = torch.tensor(D_q.ndata['x'], dtype=torch.float32)
            save_graphs(path, [D_da, D_q], graph_labels)
            i = i + 1


ppath = './数据/test/'
datadata(core_size=9, whole_size=18, n=100, ppath=ppath, self_loop=True, p=0.2)

ppath = './数据/train/'
datadata(core_size=9, whole_size=18, n=10000, ppath=ppath, self_loop=True, p=0.2)

# 检查是否唯一子图
# nx_da = test_da.to_networkx(node_attrs=['x'])
# nx_q = test_q.to_networkx(node_attrs=['x'])
# mm = DiGraphMatcher(G1=nx_da, G2=nx_q, node_match=nm)
# aa = list(mm.subgraph_isomorphisms_iter())
# if len(aa) == 1:
#     path = ppath + str(i) + '.bin'
