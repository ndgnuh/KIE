import random
from typing import Dict, Optional, Iterable, Generator
from collections import defaultdict
from functools import partial, reduce

import numpy as np
import torch
from lightning import Fabric
from torch import optim
from pydantic import BaseModel, Field
from tqdm import tqdm

from kie.models import KieModel, KieOutput, Tokenizer
from kie.data import make_dataloader, InputProcessor, Sample, EncodedSample
from kie.prettyprint import simple_postprocess as prettify_sample
from kie import processor_v2
from kie.graph_utils import ee2adj, adj2ee
from . import augments as A
from .configs import TrainConfig, ModelConfig
# augment import compose, RandomPermutation, with_probs


def loop_over_loader(loader: Iterable, n: int) -> Generator:
    """
    Returns a generator that iterates over `loader` steps by steps for `n` steps
    """
    step = 0
    while True:
        for batch in loader:
            step = step + 1
            yield step, batch
            if step >= n:
                return



class Trainer:
    def __init__(self, train_config: TrainConfig, model_config: ModelConfig):
        # Initialize model
        self.model = KieModel(model_config)
        self.tokenizer = Tokenizer(model_config)
        self.fabric = Fabric(accelerator="auto")

        # Load data
        self.processor = processor_v2.Processor(
            tokenizer=self.tokenizer,
            classes=model_config.classes
        )

        _make_dataloader = partial(
            make_dataloader,
            dataloader_options=dict(
                **train_config.dataloader,
                collate_fn=self.processor.collate_fn(),
            ),
        )
        transform_train = A.Pipeline([
            A.with_probs(A.RandomPermutation(copy=False), 0.5),
            self.processor.encode
        ])
        print(transform_train)
        transform_val = self.processor.encode
        self.train_loader = _make_dataloader(
            root=train_config.train_data,
            transform=transform_train
        )
        self.validate_loader = _make_dataloader(
            root=train_config.validate_data,
            transform=transform_val
        )

        # Check num class constrain
        # Model have +1 class for background (no class)
        assert len(self.train_loader.dataset.classes) == len(
            self.validate_loader.dataset.classes
        )
        assert len(self.train_loader.dataset.classes)\
            == (model_config.num_classes - self.processor.num_special_tokens)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=train_config.lr)
        self.lr_scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                          max_lr=train_config.lr,
                                                          pct_start=0.01,
                                                          # final_div_factor=2,
                                                          total_steps=train_config.total_steps)

        # Store configs
        self.train_config = train_config
        self.model_config = model_config

    def state_dict(self):
        # optimizer_state = self.optimizer.state_dict()
        # optimizer_state.pop("param_groups")
        # Super heavy
        # "optimizer": optimizer_state,
        # "model": self.model.state_dict(),
        return {
            "current_step": self.current_step
        }

    def train(self):
        total_steps = self.train_config.total_steps
        print_every = self.train_config.print_every
        validate_every = self.train_config.validate_every
        if print_every is None:
            print_every = max(1, validate_every // 5)

        fabric = self.fabric
        model, optimizer = self.fabric.setup(self.model, self.optimizer)
        lr_scheduler = self.lr_scheduler
        train_loader = self.fabric.setup_dataloaders(self.train_loader)

        pbar = tqdm(loop_over_loader(train_loader, total_steps),
                    total=total_steps,
                    dynamic_ncols=True)
        for step, batch in pbar:
            optimizer.zero_grad()
            output: KieOutput = model(batch)
            fabric.backward(output.loss)
            optimizer.step()
            lr_scheduler.step()
            pbar.set_description(
                f"#{step}/{total_steps} loss: {output.loss.item():.4e}")
            if step % print_every == 0:
                tqdm.write(f"LR: {lr_scheduler.get_last_lr()[0]:.2e}")

                # Checkpointing
                self.current_step = step
                torch.save(self.state_dict(), "checkpoint.pt")

            if step % validate_every == 0:
                self.validate()
                self.save_model()
                # Tested model.training, it is still true without manually set so

        # Save one last time
        self.save_model()

    @ torch.no_grad()
    def validate(self):
        model = self.fabric.setup(self.model)
        model = model.eval()
        loader = self.fabric.setup_dataloaders(self.validate_loader)

        def dict_get_index(d, i):
            return {k: v[i] for k, v in d.items()}

        post_process = self.processor.decode

        losses = []
        final_outputs = []
        metrics = defaultdict(list)

        def f1(pr, gt):
            tp = torch.count_nonzero((pr == 1) & (gt == 1))
            fp = torch.count_nonzero((pr == 1) & (gt == 0))
            # tn = (pr == 0) == (gt == 1)
            fn = torch.count_nonzero((pr == 0) & (gt == 1))
            f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
            return f1

        for batch in tqdm(loader, "validating"):
            batch_size = batch['texts'].shape[0]
            outputs: KieOutput = model(batch)
            for i in range(batch_size):
                sample = batch[i]
                # Relation scores
                metrics['rel_f1'].append(
                    f1(outputs.relations, sample.adj).cpu().item()
                )
                metrics['cls_f1'].append(
                    f1(outputs.classes, sample.classes).cpu().item()
                )

                # Extract
                sample = sample.to_numpy()
                output = outputs[i]

                # Postprocess GT
                gt = post_process(sample)

                # Postprocess PR
                sample.classes = output.classes.cpu().numpy()
                sample.adj = output.relations.cpu().numpy()
                pr = post_process(sample)

                final_outputs.append((pr, gt))
            metrics['loss'].append(outputs.loss.item())

        classes = loader.dataset.classes
        for pr, gt in random.choices(final_outputs, k=1):
            tqdm.write('PR:\t' + str(prettify_sample(pr, classes)))
            tqdm.write("+" * 3)
            tqdm.write('GT:\t' + str(prettify_sample(gt, classes)))
            tqdm.write('-' * 30)

        stats = {k: sum(v) / len(v) for k, v in metrics.items()}
        from pprint import pformat
        tqdm.write(pformat(stats))

    def save_model(self):
        self.model_save_path = "model.pt"
        torch.save(
            self.model.state_dict(),
            self.model_save_path,
        )


if __name__ == "__main__":
    from icecream import install
    from transformers import AutoTokenizer

    install()
    tokenizer_name = "vinai/phobert-base"
    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # dataloader = make_dataloader("data/inv_aug_noref_noimg.json", prepare_fn(tokenizer))
    model_config = ModelConfig(
        backbone_name="microsoft/layoutlm-base-cased",
        word_embedding_name=tokenizer_name,
        head_dims=256,
        num_classes=15,
    )

    train_config = TrainConfig(
        total_steps=1000,
        validate_every=100,
        train_data="data/inv_aug_noref_noimg.json",
        validate_data="data/inv_aug_noref_noimg.json",
        lr=5e-5,
    )
    trainer = Trainer(train_config, model_config)
    ic(trainer)
    trainer.train()
    # print(model)
    # for batch in dataloader:
    #     output = model(batch)
    #     ic(output)
    # break
    for i in loop_over_loader(range(100), 5):
        ic(i)
