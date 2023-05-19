from typing import Dict, Optional, Iterable, Generator
from functools import partial

import torch
from lightning import Fabric
from torch import optim
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from tqdm import tqdm

from kie.models import KieConfig, KieModel, KieOutput
from kie.data import make_dataloader, prepare_fn


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


class TrainConfig(BaseModel):
    # Scheduling
    total_steps: int
    validate_every: int

    # Data
    train_data: str
    validate_data: str

    # Train process
    lr: float

    # Optionals
    dataloader: Dict = Field(default_factory=dict)
    print_every: Optional[int] = None

    def __post_init__(self):
        if self.print_every is None:
            self.print_every = max(self.validate_every // 5, 1)


class Trainer:
    def __init__(self, train_config: TrainConfig, model_config: KieConfig):
        # Initialize model
        self.model = KieModel(model_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_config.word_embedding_name)
        self.fabric = Fabric(accelerator="auto")

        # Load data
        _make_dataloader = partial(
            make_dataloader,
            transform=prepare_fn(self.tokenizer),
            dataloader_options=train_config.dataloader,
        )
        self.train_loader = _make_dataloader(root=train_config.train_data)
        self.validate_loader = _make_dataloader(root=train_config.validate_data)

        # Check num class constrain
        # Model have +1 class for background (no class)
        assert len(self.train_loader.dataset.classes) == len(
            self.validate_loader.dataset.classes
        )
        assert len(self.train_loader.dataset.classes) == model_config.num_classes - 1

        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=train_config.lr)
        self.lr_scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer,
                                                          max_lr=train_config.lr,
                                                          pct_start=0.01,
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
            pbar.set_description(f"#{step}/{total_steps} loss: {output.loss.item():.4e}")
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

    @torch.no_grad()
    def validate(self):
        model = self.fabric.setup(self.model)
        model = model.eval()
        loader = self.fabric.setup_dataloaders(self.validate_loader)

        losses = []
        for batch in tqdm(loader, "validating"):
            output: KieOutput = model(batch)
            losses.append(output.loss.item())

        loss = sum(losses) / len(losses)
        tqdm.write(f"Validation loss: {loss}")

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
    model_config = KieConfig(
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
        lr=8e-5,
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
