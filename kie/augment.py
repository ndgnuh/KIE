import random
from functools import wraps, reduce, cached_property
from dataclasses import dataclass
from typing import List
from .data import Sample


def recursive_copy(obj):
    if isinstance(obj, dict):
        new_dict = {}
        for key, value in obj.items():
            new_dict[key] = recursive_copy(value)
        return new_dict
    elif isinstance(obj, list):
        return [recursive_copy(item) for item in obj]
    else:
        return obj


@dataclass
class RandomPermutation:
    copy: bool = True

    def __call__(self, sample: Sample) -> Sample:
        if self.copy:
            new_sample = recursive_copy(sample)
        else:
            new_sample = sample

        new_ids = list(range(len(sample['texts'])))
        random.shuffle(new_ids)

        for new_id, old_id in enumerate(new_ids):
            new_sample['boxes'][new_id] = sample['boxes'][old_id]
            new_sample['texts'][new_id] = sample['texts'][old_id]

        for i, pair in enumerate(sample['links']):
            new_sample['links'][i] = [new_ids.index(x) for x in pair]

        new_sample['classes'] = {}
        for key in sample['classes']:
            old_id = int(key)
            new_sample['classes'][new_ids.index(
                old_id)] = sample['classes'][key]

        return new_sample


def _compose(f, g):
    """
    return function which is composed of 2 function f g
    """
    def composed(x):
        return f(g(x))
    return composed


def compose(*functions):
    """
    return the function which is composed of functions
    """
    func = reduce(_compose, functions)
    func.__repr__ = '\n'.join([repr(f) for f in functions])
    return func


@dataclass
class Compose:
    functions: List

    @cached_property
    def composed(self):
        return compose(self.functions)

    def __call__(self, *a, **k):
        return self.composed(*a, **k)


def _with_probs(p: float):
    """
    Returns a function that call the wrapped with probability
    """
    def wrapper(f):
        @wraps(f)
        def wrapped(sample):
            if random.uniform(0, 1) <= p:
                return f(sample)
            else:
                return sample

        return wrapped
    return wrapper


def _with_probs_call(callback, p):
    """
    Same as _with_probs, but not lazy
    """
    return _with_probs(p)(callback)


def with_probs(*args, callback=None, p=None):
    """
    This is the ergonomic wrapper of `_with_probs` and `_with_probs_call`
    which puts `callback` before `p` if possible.
    """
    if callback is None and p is None:
        if len(args) == 2:
            callback, p = args
        elif len(args) == 1:
            p = args[0]
    elif callback is None and len(args) == 1:
        callback = args[0]

    if callback is None and p is not None:
        return _with_probs(p)
    elif callback is not None and p is not None:
        return _with_probs_call(callback, p)
    else:
        raise ValueError(f"Invalid inputs {args}, callback={callback}, p={p}")
