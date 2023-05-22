from dataclasses import dataclass, make_dataclass
from . import functional as F
from .functional import *


def snake_to_camel(name):
    return "".join(part.title() for part in name.split("_"))


def wrap_transform(func, *fields):
    def __call__(self, sample):
        return func(sample, **vars(self))

    Wrapped = make_dataclass(snake_to_camel(func.__name__), fields)
    Wrapped.__call__ = __call__
    return Wrapped


def wrap_hof(func, *fields):
    def get_func(self):
        f = func(**vars(self))
        return f

    def __call__(self, x):
        return self.func(x)

    Wrapped = make_dataclass(snake_to_camel(func.__name__), fields)
    Wrapped.func = property(get_func)
    Wrapped.__call__ = __call__
    return Wrapped


Compose = wrap_hof(F.compose, "functions")
Pipeline = wrap_hof(F.pipeline, "functions")
WithProbs = wrap_hof(F.with_probs, "callback", "p")
RandomPermutation = wrap_transform(F.random_permutation, "copy")
RandomRotate = wrap_transform(F.random_rotate, "min_degree", "max_degree")
