import operator as ops
from itertools import product
from pprint import pformat
from dataclasses import dataclass
from typing import *

import numpy as np

T = TypeVar("T", int, float)


@dataclass
class Metric(Generic[T]):
    mode: str = "max"
    current: T = 0
    best: T = 0

    def __post_init__(self):
        assert self.mode in ["max", "min"]
        self.best = self.init_best()
        self.current = self.best

    @property
    def should_update(self):
        return ops.le if self.mode == "max" else ops.ge

    def update(self, value):
        if np.isnan(value):
            return False
        updated = self.should_update(self.current, value)
        if updated:
            self.best = value
        self.current = value
        return updated

    def __repr__(self):
        return pformat({"current": self.current, "best": self.best})

    def init_best(self):
        if self.mode == "max":
            return -9999
        else:
            return 9999


@dataclass
class Statistics(list):
    reduce_fn: Callable

    def get(self):
        return self.reduce_fn(self)


def get_tensor_f1(pr, gt):
    import torch

    tp = torch.count_nonzero((pr == 1) & (gt == 1))
    fp = torch.count_nonzero((pr == 1) & (gt == 0))
    # tn = (pr == 0) == (gt == 1)
    fn = torch.count_nonzero((pr == 0) & (gt == 1))
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
    return f1


def get_e2e_f1(pr_list, gt_list):
    matches = []
    for (pr_cls, pr_content), (gt_cls, gt_content) in product(pr_list, gt_list):
        if pr_cls == gt_cls and gt_content == pr_content:
            matches.append((pr_cls, pr_content))
    tp = len(matches)
    fp = len(set(pr_list) - set(matches))
    fn = len(set(gt_list) - set(matches))

    f1 = (2 * tp) / (2 * tp + fn + fp + 1e-6)
    return f1
