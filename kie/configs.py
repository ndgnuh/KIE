from typing import *
from os import path
from pydantic import BaseModel, Field, validator

from .utils import read
from . import const


class TrainConfig(BaseModel):
    # Scheduling
    total_steps: int
    validate_every: int

    # Data
    train_data: Union[str, List[str]]
    validate_data: Union[str, List[str]]

    # Train process
    lr: float

    # Optionals
    dataloader: Dict = Field(default_factory=dict)
    print_every: Optional[int] = None

    @validator("print_every", always=True)
    def _print_every(cls, print_every, values):
        if print_every is None:
            print_every = max(values["validate_every"] // 5, 1)
        return print_every

    @classmethod
    def from_file(cls, file_path):
        config = read(file_path)
        return cls.parse_obj(config)


class ModelConfig(BaseModel):
    backbone: str
    classes: List[str]
    num_special_classes: int
    head_dims: int

    word_embeddings: Optional[str] = None
    pretrained_weights: Optional[str] = None
    inference_weights: Optional[str] = None
    name: Optional[str] = None

    @property
    def num_classes(self):
        return len(self.classes) + self.num_special_classes

    @classmethod
    def from_file(cls, file_path):
        config = read(file_path)
        config["name"] = path.splitext(path.basename(file_path))[0]
        return cls.parse_obj(config)

    @property
    def best_weight_path(self):
        name = f"{self.name}.{const.best_suffix}.pt"
        return path.join(const.weight_directory, name)

    @property
    def latest_weight_path(self):
        name = f"{self.name}.{const.latest_suffix}.pt"
        return path.join(const.weight_directory, name)

    @property
    def log_path(self):
        return path.join(const.log_path, self.name)
