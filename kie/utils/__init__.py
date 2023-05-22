from dataclasses import dataclass, field
from typing import *
from . import fileio
from .download import download, down_or_load
from .functional import Compose

# Work around Python behaviours
# https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses


@dataclass
class BatchNamespace:
    batched: bool = field(default=True, init=False)

    def __getitem__(self, i):
        if isinstance(i, str):
            return getattr(self, i)

        results = {}
        T = type(self)
        for k, _ in get_type_hints(self).items():
            if k == "batched":
                continue
            v = self[k]
            if not hasattr(v, "__getitem__"):
                continue
            if k in T.excluded():
                results[k] = v
            else:
                results[k] = v[i]
        ret = T(**results)
        ret.batched = False
        return ret

    @classmethod
    def excluded(cls):
        return []

    @property
    def batch_size(self):
        return len(next(iter(vars(self).values()))[0])

    def __iter__(self):
        for i in range(self.batch_size):
            yield self[i]

    # Dict-ish interface
    def items(self):
        d = vars(self)
        if "batched" in d:
            d.pop("batched")
        return d.items()

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        # Loosely
        return hasattr(self, key)


def load_pt(path):
    """
    Wrapper around `torch.load` that allows downloading files.
    """
    import torch
    path = down_or_load(path)
    return torch.load(path, map_location='cpu')


def read(path: str):
    """
    Wrapper around `utils.fileio.read` that allows downloading files.
    """
    path = down_or_load(path)
    return fileio.read(path)
