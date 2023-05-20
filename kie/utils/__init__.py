from .fileio import read
from .functional import Compose
from dataclasses import dataclass
from typing import *


@dataclass
class BatchNamespace:
    def __getitem__(self, i):
        if isinstance(i, str):
            return getattr(self, i)

        T = type(self)
        results = {}
        for k, _ in get_type_hints(self).items():
            v = self[k]
            if not hasattr(v, "__getitem__"):
                continue
            if k in T.excluded():
                results[k] = v
            else:
                results[k] = v[i]
        return T(**results)

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
        return vars(self).items()

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __contains__(self, key):
        # Loosely
        return hasattr(self, key)
