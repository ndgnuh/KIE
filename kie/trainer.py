import random
import os
from pprint import pformat
from dataclasses import dataclass
from typing import Dict, Optional, Iterable, Generator
from collections import defaultdict
from functools import partial, reduce

import numpy as np
import torch
from lightning import Fabric
from torch import nn
from torch import optim
from pydantic import BaseModel, Field
from tqdm import tqdm

from . import processor_v2
from . import augments as A
from . import utils
from .models import KieModel, KieOutput, Tokenizer
from .data import make_dataloader, InputProcessor, Sample, EncodedSample
from .prettyprint import simple_postprocess as prettify_sample
from .configs import TrainConfig, ModelConfig
from .metrics import Metric, Statistics, get_tensor_f1, get_e2e_f1

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


@dataclass
class TrainingMetrics:
    f1_classification: Metric = Metric(mode="max")
    f1_relations: Metric = Metric(mode="max")
    f1_end2end: Metric = Metric(mode="max")
    validation_loss: Metric = Metric(mode="min")
    training_loss: Metric = Metric(mode="min")
    lr: float = 0


class Trainer:
    def __init__(self, train_config: TrainConfig, model_config: ModelConfig):
        # Initialize model
        self.model = KieModel(model_config)
        self.tokenizer = Tokenizer(model_config)
        self.fabric = Fabric(accelerator="auto")
        try:
            weights = utils.load_pt(model_config.pretrained_weights)
            model.load_state_dict(weights)
        except Exception as e:
            print(
                f"Can't not load pretrained weight \
                {model_config.pretrained_weights},\
                error: {e}, ignoring"
            )

        # Load data
        self.processor = processor_v2.Processor(
            tokenizer=self.tokenizer, classes=model_config.classes
        )

        _make_dataloader = partial(
            make_dataloader,
            dataloader_options=dict(
                **train_config.dataloader,
                collate_fn=self.processor.collate_fn(),
            ),
        )
        transform_train = A.Pipeline(
            [
                A.WithProbs(A.RandomPermutation(copy=False), 0.3),
                A.WithProbs(A.RandomRotate(
                    min_degree=-10, max_degree=10), 0.3),
                self.processor.encode,
            ]
        )
        print(transform_train)
        transform_val = self.processor.encode
        self.train_loader = _make_dataloader(
            root=train_config.train_data, transform=transform_train
        )
        self.validate_loader = _make_dataloader(
            root=train_config.validate_data, transform=transform_val
        )

        # Check num class constrain
        # Model have +1 class for background (no class)
        assert len(self.train_loader.dataset.classes) == len(
            self.validate_loader.dataset.classes
        )
        assert len(self.train_loader.dataset.classes) == (
            model_config.num_classes - self.processor.num_special_tokens
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=train_config.lr)
        self.lr_scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=train_config.lr,
            pct_start=0.01,
            # final_div_factor=2,
            total_steps=train_config.total_steps,
        )

        # Store configs
        self.train_config = train_config
        self.model_config = model_config

        # Metrics
        self.metrics = TrainingMetrics(lr=train_config.lr)

    def state_dict(self):
        # optimizer_state = self.optimizer.state_dict()
        # optimizer_state.pop("param_groups")
        # Super heavy
        # "optimizer": optimizer_state,
        # "model": self.model.state_dict(),
        return {"current_step": self.current_step}

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

        pbar = tqdm(
            loop_over_loader(train_loader, total_steps),
            total=total_steps,
            dynamic_ncols=True,
        )

        train_loss = Statistics(np.mean)
        for step, batch in pbar:
            optimizer.zero_grad()
            output: KieOutput = model(batch)
            fabric.backward(output.loss)
            fabric.clip_gradients(model, optimizer, max_norm=5)
            optimizer.step()
            lr_scheduler.step()
            pbar.set_description(
                f"#{step}/{total_steps} loss: {output.loss.item():.4e}"
            )

            train_loss.append(output.loss.item())
            self.metrics.lr = lr_scheduler.get_last_lr()[0]
            if step % print_every == 0:
                self.metrics.training_loss.update(train_loss.get())

                # Checkpointing
                self.current_step = step
                self.save_model(self.model_config.latest_weight_path)
                train_loss = Statistics(np.mean)

            if step % validate_every == 0:
                self.validate()

        # Save one last time
        self.save_model()

    @ torch.no_grad()
    def validate(self, loader=None):
        model = self.fabric.setup(self.model)
        model = model.eval()
        loader = self.fabric.setup_dataloaders(loader or self.validate_loader)

        def dict_get_index(d, i):
            return {k: v[i] for k, v in d.items()}

        post_process = self.processor.decode

        losses = []
        final_outputs = []
        metrics = defaultdict(list)

        metrics = {k: Statistics(np.mean) for k in vars(self.metrics).keys()}
        for batch in tqdm(loader, "validating"):
            batch_size = batch["texts"].shape[0]
            outputs: KieOutput = model(batch)
            for i in range(batch_size):
                sample = batch[i]

                # Relation scores
                score = get_tensor_f1(
                    outputs.relations, sample.adj).cpu().item()
                metrics["f1_relations"].append(score)

                # Classification score
                score = get_tensor_f1(
                    outputs.classes, sample.classes).cpu().item()
                metrics["f1_classification"].append(score)

                # Extract
                sample = sample.to_numpy()
                output = outputs[i]

                # Postprocess GT
                gt = post_process(sample)

                # Postprocess PR
                sample.classes = output.classes.cpu().numpy()
                sample.adj = output.relations.cpu().numpy()
                pr = post_process(sample)

                # End to end format
                pr = prettify_sample(pr, self.model_config.classes)
                gt = prettify_sample(gt, self.model_config.classes)

                # End to end score
                score = get_e2e_f1(pr, gt)
                metrics["f1_end2end"].append(score)

                final_outputs.append((pr, gt))
            metrics["validation_loss"].append(outputs.loss.item())

        for pr, gt in random.choices(final_outputs, k=1):
            tqdm.write("PR:\t" + str(pr))
            tqdm.write("+" * 3)
            tqdm.write("GT:\t" + str(gt))
            tqdm.write("-" * 30)

        f1_end2end = metrics.pop("f1_end2end")
        if self.metrics.f1_end2end.update(f1_end2end.get()):
            self.save_model(self.model_config.best_weight_path)

        for k, v in metrics.items():
            metric = getattr(self.metrics, k)
            if isinstance(metric, Metric):
                metric.update(v.get())

        tqdm.write(pformat(vars(self.metrics)))

    def save_model(self, save_path):
        dirname = os.path.dirname(save_path)
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.model.state_dict(), save_path)
        tqdm.write(f"Model saved to {save_path}")
