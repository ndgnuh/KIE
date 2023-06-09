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
        should_update = self.should_update(self.current, value)
        if should_update:
            self.best = value
        self.current = value
        return should_update

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
    remove_none: bool = True

    def get(self):
        if self.remove_none:
            return self.reduce_fn([v for v in self if v is not None])
        else:
            return self.reduce_fn(self)


def get_tensor_f1(pr, gt):
    tp = ((pr == 1) & (gt == 1)).sum()
    fp = ((pr == 1) & (gt == 0)).sum()
    fn = ((pr == 0) & (gt == 1)).sum()
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


def get_e2e_f1_per_class(pr_list, gt_list, classes):
    matches = []
    for (pr_cls, pr_content), (gt_cls, gt_content) in product(pr_list, gt_list):
        if pr_cls == gt_cls and gt_content == pr_content:
            matches.append((pr_cls, pr_content))
    tps = matches
    fps = set(pr_list) - set(matches)
    fns = set(gt_list) - set(matches)

    f1s = {}
    for c in classes:
        tps_c = [pair for pair in tps if pair[0] == c]
        fps_c = [pair for pair in fps if pair[0] == c]
        fns_c = [pair for pair in fns if pair[0] == c]
        tp = len(tps_c)
        fp = len(fps_c)
        fn = len(fns_c)
        # There is no pairs
        if tp == 0 and fp == 0 and fn == 0:
            f1s[c] = None
        else:
            f1 = (2 * tp) / (2 * tp + fn + fp + 1e-6)
        f1s[c] = f1

    return f1s
