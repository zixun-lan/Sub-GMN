import numpy as np
import dgl
import torch
import torch.nn as nn
import torch
from dgl.nn.pytorch.conv import GraphConv


def to_predict_matching(raw_predict_matching):  # raw_predict_matching  bx5x18  np.array
    shape = raw_predict_matching.shape
    zeros = np.zeros(shape)
    dim2 = np.arange(shape[1])
    dim3 = np.argmax(raw_predict_matching, 2)
    for i in np.arange(shape[0]):
        zeros[i, dim2, dim3[i]] = 1
    m = zeros
    return m


def acc_renzao(predict_matching, q_size, da_size):
    shape = predict_matching.shape
    predict_matching = np.sum(predict_matching, 1)
    predict_matching = predict_matching >= 1
    predict_matching = predict_matching.astype(float).reshape(-1)

    label = np.zeros(da_size)
    label[0:q_size] = 1
    label = torch.tensor(label)
    label = label.repeat(shape[0])
    label = label.numpy()

    acc = predict_matching == label
    acc = acc.astype(float)
    acc = np.mean(acc)
    return acc
