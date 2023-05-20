import numpy as np
from typing import get_type_hints
from dataclasses import dataclass
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from .data import Sample, EncodedSample


def inter(a, b, pct):
    return a * pct + (1 - pct) * b


def mode(x, default):
    if len(x) == 0:
        return default
    modals, counts = np.unique(x, return_counts=True)
    index = np.argmax(counts)
    return modals[index]


def tokenize_texts(tokenizer, texts):
    tokens = []
    position_ids = []
    num_tokens = []
    for idx, text in enumerate(texts):
        token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        num_tokens_ = len(token)
        num_tokens.append(num_tokens_)
        position_ids.extend([idx] * num_tokens_)
        tokens.extend(token)
    return tokens, position_ids, num_tokens


def detokenize_texts(tokenizer, text_tokens, num_tokens):
    idx = 0
    decoded = []
    for num_token in num_tokens:
        text = tokenizer.decode([text_tokens[idx + i]
                                 for i in range(num_token)])
        idx = idx + num_token
        decoded.append(text)
    return decoded


def tokenize_boxes(num_tokens, boxes):
    token_boxes = [[box] * num_tokens[i] for (i, box) in enumerate(boxes)]
    token_boxes = [box for nested_box in token_boxes for box in nested_box]
    return token_boxes


def tokenize_boxes(num_tokens, boxes):
    encoded = []

    for (n, box) in zip(num_tokens, boxes):
        x0, y0 = box[0]
        x1, y1 = box[1]
        x2, y2 = box[2]
        x3, y3 = box[3]

        xt = np.linspace(x0, x1, n+1)
        xb = np.linspace(x3, x2, n+1)
        yt = np.linspace(y0, y1, n+1)
        yb = np.linspace(y3, y2, n+1)

        for i in range(n):
            token_box = [(xt[i], yt[i]), (xt[i+1], yt[i+1]),
                         (xb[i+1], yb[i+1]), (xb[i], yb[i])]
            encoded.append(token_box)
    return encoded


def detokenize_boxes(num_tokens, token_boxes):
    decoded = []
    idx = 0
    for num_token in num_tokens:
        first_token = token_boxes[idx]
        last_token = token_boxes[idx + num_token - 1]
        box = [first_token[0], last_token[1], first_token[2], last_token[3]]
        # box = [first_token[0], last_token[1], first_token[2], last_token[3]]
        # box = last_token
        decoded.append(box)
        idx = idx + num_token
    decoded = np.stack(decoded, 0)
    return decoded.tolist()


def tokenize_classes(num_tokens, classes, ignore_class=0):
    token_classes = [
        [cls] + [cls] * (num_tokens[i] - 1) for (i, cls) in enumerate(classes)
    ]
    token_classes = [cls for nested in token_classes for cls in nested]
    return token_classes


def detokenize_classes(num_tokens, token_classes, ignore_class=0):
    classes = []
    idx = 0
    for num_token in num_tokens:
        _token_classes = [token_classes[i + idx] for i in range(num_token)]
        _classes = mode([c for c in _token_classes if c != 0], 0)
        classes.append(_classes)
        idx = idx + num_token
    return classes


def tokenize_links(position_ids, links):
    position_ids = np.array(position_ids)
    n = len(position_ids)
    encoded = []
    # intra links
    for mask in np.unique(position_ids):
        token_ids = np.where(position_ids == mask)
        encoded.extend([(i, j) for i, j in zip(token_ids, token_ids[1:])])

    # inter links
    for node_i, node_j in links:
        token_i = [i for i in range(n) if position_ids[i] == node_i]
        token_j = [i for i in range(n) if position_ids[i] == node_j]
        edge = (token_i[-1], token_j[0])
        encoded.append(edge)
    return encoded


def detokenize_links(position_ids, relations):
    token_links = zip(*np.where(relations))
    links = []
    for token_i, token_j in token_links:
        node_i = position_ids[token_i]
        node_j = position_ids[token_j]
        node_i != node_j and links.append((node_i, node_j))
    return links


def tokenize(tokenizer, sample: Sample) -> EncodedSample:
    token_texts, position_ids, num_tokens = tokenize_texts(
        tokenizer, sample.texts)
    token_boxes = tokenize_boxes(num_tokens, sample.boxes)
    token_classes = tokenize_classes(num_tokens, sample.list_classes())
    token_links = tokenize_links(position_ids, sample.links)

    # Convert relations to adj
    n = len(token_texts)
    relations = np.zeros((n, n)).astype(int)
    for i, j in token_links:
        relations[i, j] = 1

    encoded = EncodedSample(
        texts=token_texts,
        boxes=token_boxes,
        classes=token_classes,
        relations=relations,
        num_tokens=num_tokens,
        position_ids=position_ids,
        image_width=sample.image_width,
        image_height=sample.image_height,
    )
    return encoded


def detokenize(tokenizer, encoded: EncodedSample) -> Sample:
    texts = detokenize_texts(
        tokenizer,
        encoded["texts"],
        encoded["num_tokens"]
    )
    boxes = detokenize_boxes(
        num_tokens=encoded["num_tokens"],
        token_boxes=encoded["boxes"]
    )
    classes = detokenize_classes(
        num_tokens=encoded["num_tokens"],
        token_classes=encoded["classes"]
    )
    links = detokenize_links(
        position_ids=encoded["position_ids"],
        relations=encoded.relations
    )

    classes = {k: v - 1 for k, v in enumerate(classes) if v > 0}
    return Sample(
        texts=texts,
        boxes=boxes,
        classes=classes,
        links=links,
        image_width=encoded.image_width,
        image_height=encoded.image_height,
    )


if __name__ == "__main__":
    from tqdm import tqdm
    from kie.data import (
        KieDataset,
        prepare_input,
        make_dataloader,
        InputProcessor,
        Sample,
    )

    tokenizer_ = AutoTokenizer.from_pretrained(
        "vinai/phobert-base", local_files_only=True
    )
    root = "data/inv_aug_noref_noimg.json"
    base_dataset = KieDataset(root)
    for sample in tqdm(base_dataset):
        run_tests(tokenizer_, sample)
    encoded = tokenize(tokenizer_, sample)
    # print(encoded)
