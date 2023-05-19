import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from kie.data import (
    KieDataset,
    prepare_input,
    make_dataloader,
    InputProcessor,
    Sample
)

tokenizer_ = AutoTokenizer.from_pretrained("vinai/phobert-base", local_files_only=True)
root = "data/inv_aug_noref_noimg.json"


def test_tokenization():
    from kie.tokenize import tokenize, detokenize

    def eq(a, b):
        return np.all(np.array(a) == np.array(b))

    def run_tests(tokenizer, sample: Sample):
        encoded = tokenize(tokenizer, sample)
        texts, boxes, classes, links = detokenize(tokenizer, encoded)
        assert len(encoded["texts"]) == len(encoded["boxes"])
        assert len(encoded["texts"]) == len(encoded["classes"])
        assert eq(texts, sample.texts) or "<unk>" in "".join(texts)
        assert eq(boxes, sample.boxes)
        assert eq(classes, sample.list_classes())
        assert set(links) == set(sample.links)

    base_dataset = KieDataset(root)
    for sample in tqdm(base_dataset):
        run_tests(tokenizer_, sample)
