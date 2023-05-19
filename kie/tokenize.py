import numpy as np
from kie.processing import post_process
from kie.data import KieDataset, prepare_input, make_dataloader, InputProcessor
from transformers import AutoTokenizer
from kie.tokenize import Tokenizer


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
        text = tokenizer.decode([text_tokens[idx + i] for i in range(num_token)])
        idx = idx + num_token
        decoded.append(text)
    return decoded


def tokenize_boxes(num_tokens, boxes):
    token_boxes = [[box] * num_tokens[i] for (i, box) in enumerate(boxes)]
    token_boxes = [box for nested_box in token_boxes for box in nested_box]
    return token_boxes


def detokenize_boxes(num_tokens, token_boxes):
    decoded = []
    idx = 0
    for num_token in num_tokens:
        decoded.append(token_boxes[idx])
        idx = idx + num_token
    return decoded


def tokenize_classes(num_tokens, classes, ignore_class=0):
    token_classes = [
        [cls] + [ignore_class] * (num_tokens[i] - 1) for (i, cls) in enumerate(classes)
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
    n = len(position_ids)
    encoded = []
    for node_i, node_j in links:
        token_i = [i for i in range(n) if position_ids[i] == node_i]
        token_j = [i for i in range(n) if position_ids[i] == node_j]
        edge = (token_i[-1], token_j[0])
        encoded.append(edge)
    return encoded


def detokenize_links(position_ids, token_links):
    links = []
    for token_i, token_j in token_links:
        node_i = position_ids[token_i]
        node_j = position_ids[token_j]
        node_i != node_j and links.append((node_i, node_j))
    return links


def tokenize(tokenizer, sample):
    token_texts, position_ids, num_tokens = tokenize_texts(tokenizer, sample.texts)
    token_boxes = tokenize_boxes(num_tokens, sample.boxes)
    token_classes = tokenize_classes(num_tokens, sample.list_classes())
    token_links = tokenize_links(position_ids, sample.links)

    encoded = dict(
        texts=token_texts,
        boxes=token_boxes,
        classes=token_classes,
        links=token_links,
        num_tokens=num_tokens,
        position_ids=position_ids
    )
    return encoded


def detokenize(tokenizer, encoded):
    texts = detokenize_texts(tokenizer, encoded["texts"], encoded["num_tokens"])
    boxes = detokenize_boxes(
        num_tokens=encoded["num_tokens"], token_boxes=encoded["boxes"]
    )
    classes = detokenize_classes(
        num_tokens=encoded["num_tokens"], token_classes=encoded["classes"]
    )
    links = detokenize_links(
        position_ids=encoded["position_ids"], token_links=encoded["links"]
    )
    return texts, boxes, classes, links


def eq(a, b):
    return np.all(np.array(a) == np.array(b))


def run_tests(tokenizer, sample):
    encoded = tokenize(tokenizer, sample)
    texts, boxes, classes, links = detokenize(tokenizer, encoded)
    assert len(encoded["texts"]) == len(encoded["boxes"])
    assert len(encoded["texts"]) == len(encoded["classes"])
    assert eq(texts, sample.texts) or '<unk>' in ''.join(texts)
    assert eq(boxes, sample.boxes)
    assert eq(classes, sample.list_classes())
    assert set(links) == set(sample.links)


if __name__ == "__main__":
    from tqdm import tqdm
    tokenizer_ = AutoTokenizer.from_pretrained("vinai/phobert-base", local_files_only=True)
    root = "data/inv_aug_noref_noimg.json"
    base_dataset = KieDataset(root)
    for sample in tqdm(base_dataset):
        run_tests(tokenizer_, sample)
