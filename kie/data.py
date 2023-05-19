from functools import partial
from typing import Callable, List, Dict, Optional, Tuple, Set
from dataclasses import dataclass

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from transformers import AutoTokenizer

from .fileio import read


Point = Tuple[float, float]
Polygon = Tuple[Point, Point, Point, Point]


# Doesnt work with DataLoader
# @dataclass
# class TorchSample(dict):
#     texts: torch.Tensor
#     boxes: torch.Tensor
#     links: torch.Tensor
#     classes: torch.Tensor
#     masks: torch.Tensor
#     def items(self):
#         return vars(self).items()


class Sample(BaseModel):
    texts: List[str]
    boxes: List[Polygon]
    image_width: int
    image_height: int
    links: Set[Tuple[int, int]] = Field(default_factory=set)
    classes: Dict[int, int] = Field(default_factory=dict)


def idendity(x):
    return x


def prepare_input(tokenizer, sample: Sample):
    texts = sample.texts
    boxes = sample.boxes
    classes = sample.classes
    links = sample.links
    num_nodes = len(texts)

    #
    # 1D Tokenization (flat properties)
    #
    token_texts = []
    token_boxes = []
    token_classes = []
    token_masks = []
    for idx, (text, box) in enumerate(zip(texts, boxes)):
        # tokenize
        text_tokens = tokenizer.tokenize(text)
        num_tokens = len(text_tokens)

        # Extend other props to token length
        text_tokens = tokenizer.convert_tokens_to_ids(text_tokens)
        box_tokens = num_tokens * [box]
        class_tokens = num_tokens * [classes.get(idx, -1) + 1]
        mask_tokens = num_tokens * [idx]

        # Append
        token_texts.extend(text_tokens)
        token_boxes.extend(box_tokens)
        token_classes.extend(class_tokens)
        token_masks.extend(mask_tokens)
    #
    # Convert some to numpy and normalize
    #
    token_masks = np.array(token_masks)
    token_boxes = np.array(token_boxes, dtype="float32")
    token_boxes[..., 0] = token_boxes[..., 0] / sample.image_width
    token_boxes[..., 1] = token_boxes[..., 1] / sample.image_height

    #
    # 2D Tokenization
    #
    token_links = []
    # Link between tokens of the same boxes
    for ti, tj in zip(token_masks, token_masks[1:]):
        token_links.append((ti, tj))
    # Link assigned from node i -> node j
    for i, j in links:
        # last token of i
        (ti,) = np.where(token_masks == i)
        ti = ti[-1]
        # first token of j
        (tj,) = np.where(token_masks == i)
        tj = tj[0]
        token_links.append((ti, tj))

    #
    # Links to relations
    #
    num_tokens = len(token_texts)
    token_relations = np.zeros((num_tokens, num_tokens), dtype='long')
    for (ti, tj) in token_links:
        token_relations[ti, tj] = 1

    # To torch land, drop the `token_` prefixes
    ret = dict(
        texts=torch.tensor(token_texts),
        boxes=torch.tensor(token_boxes),
        masks=torch.tensor(token_masks),
        classes=torch.tensor(token_classes),
        relations=torch.tensor(token_relations),
    )
    return ret


class KieDataset(Dataset):
    def __init__(self, root, transform=idendity):
        super().__init__()
        data = read(root)
        self.root = root
        self.transform = transform
        self.classes = data["classes"]
        self.samples = [Sample.parse_obj(sample) for sample in data["samples"]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample: Sample = self.samples[idx]
        sample = self.transform(sample)
        return sample


def collate_fn(pad_config: Dict, samples: List):
    #
    # Collect the max shapes of each tensor type
    #
    max_sizes = dict()
    for sample in samples:
        for k, v in sample.items():
            shape = torch.tensor(v.shape)
            if k not in max_sizes:
                max_sizes[k] = shape
            else:
                max_sizes[k] = torch.maximum(max_sizes[k], shape)
    max_sizes = {k: torch.Size(v) for k, v in max_sizes.items()}

    #
    # Pad to max sizes
    #
    def pad_to_shape(x, shape, value):
        # padding_masks = torch.zeros_like(x)
        if x.shape == shape:
            return x
        padder = [[0, s2 - s1] for s1, s2 in zip(x.shape, shape)]
        padder = tuple(sum(reversed(padder), []))  # Merge list
        padded = F.pad(x, padder, value=value)
        return padded

    #
    # Stack to batch
    #
    batch = dict()
    for k, shape in max_sizes.items():
        pad_value = pad_config.get(k, None)
        batch[k] = torch.stack(
            [pad_to_shape(sample[k], shape, pad_value) for sample in samples], dim=0
        )
    return batch


def make_dataloader(root, transform, dataloader_options: Dict = dict()):
    dataset = KieDataset("./data/inv_aug_noref_noimg.json", transform=transform)
    # TODO: do not hard code this
    pad_config = dict(texts=1)
    dataloader = DataLoader(
        dataset, batch_size=4, collate_fn=partial(collate_fn, pad_config)
    )
    return dataloader


def prepare_fn(tokenizer):
    transform = partial(prepare_input, tokenizer)
    return transform


if __name__ == "__main__":
    from icecream import ic

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    loader = make_dataloader("./data/inv_aug_noref_noimg.json", transform=prepare_fn("vinai/phobert-base"))
    ic(next(iter(loader)))
    #     ic(batch)
    # dl = DataLoader(dataset, batch_size=2)