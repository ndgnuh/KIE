from typing import List
import numpy as np
import torch
from torch.nn import functional as F


def adj2ee(adj):
    return list(zip(*np.where(adj)))


def ee2adj(links, n=None):
    n = n or max(max(i, j) for i, j in links)
    adj = np.zeros(n, n).astype("bool")
    for i, j in links:
        adj[i, j] = 1
    return adj


class path_graph:
    @staticmethod
    def encode_adj(adj):
        adj = F.pad(adj, (1, 0, 1, 0))
        return torch.argmax(adj, dim=1)

    @staticmethod
    def decode_paths(paths):
        print(paths.shape)
        adj = F.one_hot(paths, paths.shape[0])
        return adj[1:, 1:]

    @staticmethod
    def decode_path_logits(paths_logits):
        probs = torch.softmax(paths_logits, dim=-1)
        scores, paths = probs.max(dim=-1)
        adj = F.one_hot(paths, paths.shape[0])
        return adj[1:, 1:]


# class generic_graph:
#     @staticmethod
#     def decode_logits(scores):
