import random
from functools import wraps, reduce
from typing import List

import numpy as np
from ..data import Sample


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


def center_transform_polygon(polygon: list[list[int, int]], transformation):
    # Numpy is easier
    polygon_np = np.array(polygon)
    # Calculate center
    center = polygon_np.mean(axis=0)
    # Translate to center
    translated_polygon = polygon_np - center
    # Transform
    rotated_polygon = np.matmul(translated_polygon, transformation)
    # Translate back
    rotated_polygon += center
    # Back to list
    return rotated_polygon.tolist()


def random_rotate(sample: Sample, min_degree: float, max_degree: float):
    deg = random.uniform(min_degree, max_degree)
    rad = np.deg2rad(deg)
    s = np.sin(rad)
    c = np.cos(rad)
    m = np.array([[c, -s], [s, c]])
    sample = sample.dict()
    sample["boxes"] = center_transform_polygon(sample["boxes"], m)
    return Sample(**sample)


def random_permutation(sample: Sample, copy: bool = True) -> Sample:
    if copy:
        new_sample = recursive_copy(sample.dict())
    else:
        new_sample = sample.dict()

    new_ids = list(range(len(sample["texts"])))
    random.shuffle(new_ids)

    for new_id, old_id in enumerate(new_ids):
        new_sample["boxes"][new_id] = sample["boxes"][old_id]
        new_sample["texts"][new_id] = sample["texts"][old_id]

    links = set()
    for i, j in sample.links:
        i2 = new_ids.index(i)
        j2 = new_ids.index(i)
        links.add((i2, j2))
    new_sample["links"] = links

    new_sample["classes"] = {}
    for key in sample["classes"]:
        old_id = int(key)
        new_sample["classes"][new_ids.index(old_id)] = sample["classes"][key]

    return Sample(**new_sample)


def _compose(f, g):
    """
    return function which is composed of 2 function f g
    """

    def composed(x):
        return f(g(x))

    return composed


def compose(functions):
    """
    return the function which is composed of functions
    """
    func = reduce(_compose, functions)
    func.__repr__ = lambda: "\n".join([repr(f) for f in functions])
    func.__str__ = lambda: "\n".join([repr(f) for f in functions])
    return func


def pipeline(functions):
    """
    return the function which is composed of functions
    """
    func = reduce(_compose, reversed(functions))
    return func


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
