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

tokenizer_ = AutoTokenizer.from_pretrained(
    "vinai/phobert-base", local_files_only=True)
root = "data/inv_aug_noref_noimg.json"


def test_tokenization():
    from kie.tokenize import tokenize, detokenize

    def eq(a, b):
        return np.all(np.array(a) == np.array(b))

    def dict_eq(a, b):
        if a.keys() != b.keys():
            return False
        return all(a[k] == b[k] for k in a)

    def box_approx(x, y):
        x = np.array(x)
        y = np.array(y)
        for x_i, y_i in zip(x, y):
            if not np.all(x_i == y_i):
                print("____")
                print(x_i)
                print(y_i)
        # print(x.shape, y.shape)
        # print(np.max(np.array(x) - np.array(y)))
        # return np.max(np.array(x) - np.array(y)) < 10
        pass

    def run_tests(tokenizer, sample: Sample):
        encoded = tokenize(tokenizer, sample)
        decoded = detokenize(tokenizer, encoded)
        print(np.array(decoded.boxes))
        print(np.array(sample.boxes))
        assert len(encoded["texts"]) == len(encoded["boxes"])
        assert len(encoded["texts"]) == len(encoded["classes"])
        assert eq(decoded.texts, sample.texts) or \
            "<unk>" in "".join(decoded.texts)
        assert box_approx(decoded.boxes, sample.boxes)
        assert dict_eq(decoded.classes, sample.classes)
        assert set(decoded.links) == set(sample.links)

    base_dataset = KieDataset(root)
    for sample in tqdm(base_dataset):
        run_tests(tokenizer_, sample)
