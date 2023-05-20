from typing import List, TypeVar, ClassVar, Any
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property

import numpy as np
from scipy.stats import mode

from kie.data import KieDataset, Sample


@dataclass
class Encoded:
    texts: np.ndarray
    boxes: np.ndarray
    classes: np.ndarray
    adj: np.ndarray
    token2item: np.ndarray

    def to_tensor(self):
        import torch
        return Encoded(**{k: torch.tensor(v) for k, v in vars(self).items()})


@dataclass
class CollateFn:
    pass


@dataclass
class Processor:
    tokenizer: Any
    classes: List[str]
    special_classes: ClassVar[List[str]] = ["<pad>", "<sub>", "<other>"]
    pad_class_id: ClassVar[int] = 0
    sub_class_id: ClassVar[int] = 1
    other_class_id: ClassVar[int] = 2

    @cached_property
    def num_special_tokens(self):
        return len(self.special_classes)

    def encode(self, sample: Sample) -> Encoded:
        n = len(sample.texts)
        texts = []
        boxes = []
        classes = []
        token2item = []
        count = 0

        #
        # Simple tokenizations
        #
        for text, box in zip(sample.texts, sample.boxes):
            class_ = sample.classes.get(
                count, self.other_class_id - self.num_special_tokens
            ) + self.num_special_tokens

            # Tokenize
            text_tokens = tokenizer.tokenize(text)
            text_tokens = tokenizer.convert_tokens_to_ids(text_tokens)
            num_tokens = len(text_tokens)

            # Extend accordingly
            texts.extend(text_tokens)
            boxes.extend([box] * num_tokens)
            classes.extend([class_] + [self.sub_class_id] * (num_tokens - 1))
            token2item.extend([count] * num_tokens)
            count = count + 1

        #
        # Build a mapping
        #
        item2token = defaultdict(list)
        for token_idx, item_idx in enumerate(token2item):
            item2token[item_idx].append(token_idx)

        #
        # Convert links to tokens
        #
        links = []
        # intra links
        for tokens in item2token.values():
            links.extend(zip(tokens, tokens[1:]))
        for i, j in sample.links:
            # inter links
            ni = item2token[i]
            nj = item2token[j]
            links.append((ni[-1], nj[0]))

            # convert linked tokens to the sub classes
            for idx in nj:
                classes[idx] = self.sub_class_id

        # Normalize bounding boxes
        boxes = np.array(boxes) * 1.0
        boxes[:, :, 0] /= sample.image_width
        boxes[:, :, 1] /= sample.image_height

        # convert links to adj
        N = len(texts)
        adj = np.zeros((N, N), dtype=int)
        for i, j in links:
            adj[i, j] = 1

        return Encoded(
            texts=np.array(texts),
            boxes=boxes,
            classes=np.array(classes),
            adj=adj,
            token2item=np.array(token2item)
        )

    def decode(self, encoded: Encoded) -> Sample:
        token2item = encoded.token2item
        texts = []
        boxes = []
        classes = dict()
        n = len(encoded.texts)

        # Simple decode
        for item_id in np.unique(token2item):
            mask = token2item == item_id

            texts.append(tokenizer.decode(encoded.texts[mask]))
            boxes.append(encoded.boxes[mask][0].tolist())
            try:
                _classes = encoded.classes[mask] - self.num_special_tokens
                _classes = _classes[_classes >= 0]
                if len(_classes) == 0:
                    continue
                classes[item_id] = mode(_classes, keepdims=False).mode
            except IndexError:
                pass  # mode of empty

        # links from adj
        links = []
        starts, ends = np.where(encoded.adj)
        for i, j in zip(starts, ends):
            ni = encoded.token2item[i]
            nj = encoded.token2item[j]
            if ni != nj:
                links.append((ni, nj))

        return Sample(
            texts=texts,
            boxes=boxes,
            links=links,
            classes=classes,
            image_width=0,
            image_height=0,
        )

    def collate_fn(self) -> CollateFn:
        pass


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    dataset = KieDataset("data/val.json")
    processor = Processor(tokenizer=tokenizer,
                          classes=dataset.classes)
    sample = dataset[0]
    enc = processor.encode(sample)
    dec = processor.decode(enc)

    # print(enc.classes)
    # print(sample.classes)
    # print(dec.classes)
    assert set(sample.texts) == set(dec.texts)
    assert set(sample.links) == set(dec.links)
    assert set(sample.classes.items()) == set(dec.classes.items())
