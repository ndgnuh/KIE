from dataclasses import dataclass
from typing import *


@dataclass
class Compose:
    callables: List[Callable]

    def __call__(self, x, *a, **kwargs):
        for function in self.callables:
            x = function(x, *a, **kwargs)
        return x

