from typing import List
import numpy as np


def adj2ee(adj):
    return list(zip(*np.where(adj)))


def ee2adj(links, n=None):
    n = n or max(max(i, j) for i, j in links)
    adj = np.zeros(n, n).astype("bool")
    for i, j in links:
        adj[i, j] = 1
    return adj
