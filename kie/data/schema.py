from typing import *
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

import random
from copy import deepcopy

def enforce_convert(cls):
    th = get_type_hints(cls)

    def __post_init__(self):
        for k, f in th.items():
            setattr(self, k, f(getattr(self, k)))

    cls.__post_init__ = __post_init__
    return cls


def augment(sample):
    new_sample = deepcopy(sample)
    
    new_ids = list(range(len(sample['texts'])))
    random.shuffle(new_ids)

    
    for new_id, old_id in enumerate(new_ids):
        new_sample['boxes'][new_id] = sample['boxes'][old_id]
        new_sample['texts'][new_id] = sample['texts'][old_id]
    
    for i, pair in enumerate(sample['links']):
        new_sample['links'][i]=[new_ids.index(x) for x in pair]

    new_sample['classes']={}
    for key in sample['classes']:
        old_id = int(key)
        new_sample['classes'][new_ids.index(old_id)] = sample['classes'][key]
    
    return new_sample

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
