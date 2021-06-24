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
import os


class dgraph(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.graph_pairs = os.listdir(self.root_dir)

    def __getitem__(self, index):
        graph_pair_index = self.graph_pairs[index]
        graph_pair_path = os.path.join(self.root_dir, graph_pair_index)
        graph_pair, label_dict = load_graphs(graph_pair_path)
        graph_da = graph_pair[0]
        graph_q = graph_pair[1]
        label = np.array(label_dict['glabel'])
        same = np.array(label_dict['same_m'])
        return graph_da, graph_q, label, same

    def __len__(self):
        return len(self.graph_pairs)


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    g1, g2, labels, sames = map(list, zip(*samples))
    bg1 = dgl.batch(g1)
    bg2 = dgl.batch(g2)
    return bg1, bg2, torch.tensor(labels), torch.tensor(sames)
