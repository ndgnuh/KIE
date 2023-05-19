from .fileio import read
from .functional import Compose


class BatchDict(dict):
    def __post_init__(self):
        self.__dict__ = self

    def __getitem__(self, idx: int):
        if isinstance(idx, str):
            return dict.__getitem__(self, idx)
        return {idx: v[k] for k, v in self.items()}


def ez_get_item(cls):
    def __getitem__(self, idx: int):
        if isinstance(idx, str):
            return setattr(self, idx)
        result = {}
        for k, v in vars(self).items():
            try:
                result[k] = v[idx]
            except IndexError:
                result[k] = v
        return result

    cls.__getitem__ = __getitem__
    return cls
