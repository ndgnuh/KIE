from typing import *
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field


def enforce_convert(cls):
    th = get_type_hints(cls)

    def __post_init__(self):
        for k, f in th.items():
            setattr(self, k, f(getattr(self, k)))

    cls.__post_init__ = __post_init__
    return cls


class Sample(BaseModel):
    texts: List[str]
    boxes: List
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    links: Set[Tuple[int, int]] = Field(default_factory=set)
    classes: Dict[int, int] = Field(default_factory=dict)

    def list_classes(self):
        classes = [0] * len(self.texts)
        for k, c in self.classes.items():
            classes[k] = c + 1
        return classes

    def __getitem__(self, idx):
        return getattr(self, idx)


@dataclass
@enforce_convert
class EncodedSample:
    texts: np.array
    boxes: np.array
    classes: np.array
    relations: np.array
    num_tokens: np.array
    position_ids: np.array
    image_width: np.array
    image_height: np.array

    def __getitem__(self, idx):
        return getattr(self, idx)

    def items(self):
        return vars(self).items()
