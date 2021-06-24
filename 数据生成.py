import torch
import dgl
import networkx as nx
import numpy as np
from dgl.data.utils import save_graphs
from networkx.algorithms.isomorphism import GraphMatcher, DiGraphMatcher
import networkx.algorithms.isomorphism as iso


def to_m(label, q_size, g_size):  # label 是 np1d.array
    m = np.zeros([q_size, g_size])
    for i, j in enumerate(label):
        m[i][j] = 1.
    return m


def same(q_f, da_f): # 2d array np
    z = []
    for i in q_f:
        a = (i == da_f).all(axis=1).astype(float).reshape(1, -1)
        z.append(a)
    end = np.concatenate(z, axis=0)
    return end


# data_generation1 是生成 每个样本对 的core 固定的 数据的函数
# core_size 是 core图的大小
# whole_size 是 大图的大小
# n 是生成 样本对 的 数量
# F 是 core 的固定特征
# self_loop 是否加自环
def data_generation1(core_size, whole_size, n, F, ppath, self_loop=False, p=0.3):
    # nm = iso.categorical_node_match('x', 1)
    nm = iso.numerical_node_match('x', 1.0)
    E = np.eye(10)  # E是10x10的单位矩阵 表示不同特征对应的独热编码
    aaaa = False

    while aaaa is not True:  # 防止出现 孤立点
        g_q = nx.generators.random_graphs.fast_gnp_random_graph(n=core_size, p=p)
        aaaa = nx.is_connected(g_q)

    # adj_q = nx.to_numpy_matrix(g_q) # 邻接矩阵

    if self_loop:  # 是否加自环
        e = np.eye(core_size)
        adj_q = nx.to_numpy_matrix(g_q)
        adj_q = adj_q + e
    else:
        adj_q = nx.to_numpy_matrix(g_q)

    nx_q = nx.from_numpy_matrix(adj_q)  # 再转 nx 图
    D_q = dgl.DGLGraph()
    D_q.from_networkx(nx_q)
    q_feature = E[F]
    D_q.ndata['x'] = torch.tensor(torch.tensor(q_feature), dtype=torch.float32)  # 转DGL

    B = whole_size - core_size  # 合并图 的 大小

    i = 0
    while i < n:
        bbbb = False
        while bbbb is not True:  # 防止出现 孤立点
            g_B = nx.generators.random_graphs.fast_gnp_random_graph(n=B, p=p)
            bbbb = nx.is_connected(g_B)

        if self_loop:  # 是否加自环
            e = np.eye(B)
            adj_B = nx.to_numpy_matrix(g_B)
            adj_B = adj_B + e
        else:
            adj_B = nx.to_numpy_matrix(g_B)

        # 创造 最终 邻接矩阵
        lianjie = np.random.uniform(0, 1, size=[core_size, B])
        lianjie = lianjie <= p
        lianjie = lianjie.astype(float)
        lianjie_T = lianjie.T
        shang = np.concatenate([adj_q, lianjie], axis=1)
        # print(lianjie_T.shape)
        # print(adj_B.shape)
        xia = np.concatenate([lianjie_T, adj_B], axis=1)
        end = np.concatenate([shang, xia], axis=0)

        nx_da = nx.from_numpy_matrix(end)  # 转 nx 图
        D_da = dgl.DGLGraph()
        D_da.from_networkx(nx_da)
        F_B = np.random.randint(0, 10, size=B)
        F_whole = np.append(F, F_B)
        Da_feature = E[F_whole]
        D_da.ndata['x'] = torch.tensor(torch.tensor(Da_feature), dtype=torch.float32)  # 转 dgl 图

        # test_da_q
        # test_da
        test_da = dgl.DGLGraph()
        test_da.from_networkx(nx_da)
        test_da.ndata['x'] = torch.tensor(F_whole, dtype=torch.float32).reshape(-1, 1)
        # test_da
        # test_q
        test_q = dgl.DGLGraph()
        test_q.from_networkx(nx_q)
        test_q.ndata['x'] = torch.tensor(F, dtype=torch.float32).reshape(-1, 1)
        # test_q
        # test_da_q




        # 检查是否唯一子图
        nx_da = test_da.to_networkx(node_attrs=['x'])
        nx_q = test_q.to_networkx(node_attrs=['x'])
        mm = DiGraphMatcher(G1=nx_da, G2=nx_q, node_match=nm)
        aa = list(mm.subgraph_isomorphisms_iter())
        if len(aa) == 1:
            path = ppath + str(i) + '.bin'

            D_da.ndata['x'] = torch.tensor(D_da.ndata['x'], dtype=torch.float32)
            D_q.ndata['x'] = torch.tensor(D_q.ndata['x'], dtype=torch.float32)

            same_m = same(D_q.ndata['x'].numpy(), D_da.ndata['x'].numpy())
            m = to_m(label=np.arange(core_size), q_size=core_size, g_size=whole_size)

            graph_labels = {'glabel': torch.tensor(m, dtype=torch.float32), 'same_m': torch.tensor(same_m, dtype=torch.float32)}
            save_graphs(path, [D_da, D_q], graph_labels)
            i = i + 1












        # path = ppath + str(i) + '.bin'
        #
        # m = to_m(label=np.arange(core_size), q_size=core_size, g_size=whole_size)
        #
        # graph_labels = {"glabel": torch.tensor(m, dtype=torch.float32)}
        # D_da.ndata['x'] = torch.tensor(D_da.ndata['x'], dtype=torch.float32)
        # D_q.ndata['x'] = torch.tensor(D_q.ndata['x'], dtype=torch.float32)
        # save_graphs(path, [D_da, D_q], graph_labels)


