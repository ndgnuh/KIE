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


def get_dfs_sequence(start: int, links: list):
    sequence = [start]  # Khởi tạo chuỗi với số bắt đầu

    current_number = start  # Số hiện tại trong chuỗi
    found_match = True
    while found_match:
        found_match = False
        # Tìm số sau của cặp [số trước, số sau] trong input
        for pair in links:
            if pair[0] == current_number:
                next_number = pair[1]
                found_match = True
                sequence.append(next_number)  # Thêm số sau vào chuỗi
                current_number = next_number  # Cập nhật số hiện tại
                break

    return sequence

@singledispatch
def get_dfs_sequence(adj: np.ndarray, start: int):
    visited = [False for _ in adj.shape[0]]
    def recurse(visited, i):
        subs = np.where(adj[i, :])
