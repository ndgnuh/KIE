from typing import *
from dataclasses import dataclass

import torch
from pydantic import BaseModel
from torch import nn, Tensor
from transformers import AutoModel
from .utils import BatchDict, ez_get_item
from .bros import BrosModel, BrosConfig


@torch.no_grad()
def adapt_input_bros(batch):
    # B N 4 2
    boxes = batch["boxes"] * 1.0

    # Normalize
    boxes[..., 0] = boxes[..., 0] / batch['image_width']
    boxes[..., 1] = boxes[..., 1] / batch['image_height']

    # Convert to xy8
    boxes = boxes.transpose(-1, -2).flatten(-2).float()
    batch["boxes"] = boxes
    return dict(
        input_ids=batch["texts"],
        bbox=boxes,
        attention_mask=batch.get("attention_masks", None),
    )


@torch.no_grad()
def adapt_input_layoutlm(batch):
    # B N 4 2
    boxes = batch["boxes"] * 1.0

    # Normalize
    boxes[..., 0] = boxes[..., 0] * 1000 / batch['image_width']
    boxes[..., 1] = boxes[..., 1] * 1000 / batch['image_height']

    # Convert to xyxy
    maxs = boxes.max(dim=-2).values
    mins = boxes.min(dim=-2).values
    # X min Y min X max Y max
    boxes = torch.cat([mins, maxs], dim=-1)
    boxes = torch.round(boxes).type(torch.long)
    return dict(
        bbox=boxes,
        input_ids=batch["texts"],
        attention_mask=batch.get("attention_masks", None),
    )


class KieConfig(BaseModel):
    backbone_name: str
    word_embedding_name: str
    num_classes: int
    head_dims: int


@ez_get_item
@dataclass
class KieOutput:
    class_logits: Tensor
    relation_logits: Tensor
    loss: Optional[Tensor] = None

    def __post_init__(self):
        class_probs = torch.softmax(self.class_logits, dim=-1)
        self.class_scores, self.classes = torch.max(class_probs, dim=-1)

        # Post process relation logits
        relation_probs = torch.softmax(self.relation_logits, dim=-1)
        self.relation_scores, self.relations = torch.max(
            relation_probs, dim=-1)

    def __getitem__(self, idx):
        return getattr(self, idx)


class ClassificationHead(nn.Sequential):
    def __init__(self, config: KieConfig):
        super().__init__()
        self.classify = nn.Linear(config.head_dims, config.num_classes)


class RelationTaggerHead(nn.Module):
    def __init__(self, config: KieConfig):
        super().__init__()
        head_dims = config.head_dims
        self.head = nn.Linear(head_dims, head_dims)
        self.tail = nn.Linear(head_dims, head_dims * 2)

    def forward(self, hidden):
        # N M D
        head = self.head(hidden)
        tail = self.tail(hidden)
        tail = torch.stack(tail.chunk(2, dim=-1), dim=-1)

        # logits: B N N 2
        # 0 no links
        # 1 yes links
        logits = torch.einsum("bnd,bmdc->bnmc", head, tail)
        return logits


class KieLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_loss = nn.CrossEntropyLoss()
        self.r_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.1, 1.0]))

    def forward(self, pr_class_logits, gt_classes, pr_relation_logits, gt_relations):
        # Class to dim 1
        pr_class_logits = pr_class_logits.transpose(1, -1)
        pr_relation_logits = pr_relation_logits.transpose(1, -1)
        c_loss = self.c_loss(pr_class_logits, gt_classes)
        r_loss = self.r_loss(pr_relation_logits, gt_relations)
        return c_loss + r_loss


# class RelativeSpatialAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.Q = nn.Linear(768, 768)
#         self.K = nn.Linear(768, 768)
#         self.V = nn.Linear(768, 768)

#     def forward(self, x, boxes):
#         # x: B N D
#         # bboxes: B N 8
#         Q = self.Q(boxes)
#         K = self.K(boxes)
#         V = self.V(x)
#         W = torch.matmul(Q, K.transpose(-1, -2))
#         W = torch.softmax(W / 8, dim=-1)
#         ctx = torch.matmul(W, V)
#         return ctx


class KieModel(nn.Module):
    def __init__(self, config: KieConfig):
        super().__init__()
        Pretrain = BrosModel if "bros" in config.backbone_name else AutoModel
        pretrain = Pretrain.from_pretrained(config.backbone_name)
        pretrain_we = AutoModel.from_pretrained(config.word_embedding_name)
        self.adapt = adapt_input_bros if "bros" in config.backbone_name else adapt_input_layoutlm
        pretrain.embeddings.word_embeddings = pretrain_we.embeddings.word_embeddings
        self.encoder = pretrain

        # Relative attention
        self.bbox_embeddings = nn.Linear(8, 768)
        self.rel_atn = nn.MultiheadAttention(768, 8)

        #
        # neck
        #
        self.project = nn.Sequential(
            nn.Linear(768, config.head_dims), nn.Tanh())

        #
        # Prediction heads
        #
        self.classify = ClassificationHead(config)
        self.relation_tagger = RelationTaggerHead(config)

        #
        # Loss function
        #
        self.loss = KieLoss()

    def forward(self, batch):
        # Input adapt to backbone
        inputs = self.adapt(batch)

        # Forward backbone
        hidden = self.encoder(**inputs).last_hidden_state

        hidden = self.project(hidden)

        class_logits = self.classify(hidden)
        relation_logits = self.relation_tagger(hidden)

        if "classes" in batch:
            loss = self.loss(
                class_logits, batch["classes"], relation_logits, batch["relations"]
            )
        else:
            loss = None

        return KieOutput(
            class_logits=class_logits, relation_logits=relation_logits, loss=loss
        )


if __name__ == "__main__":
    from icecream import install

    install()
    tokenizer_name = "vinai/phobert-base"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataloader = make_dataloader(
        "data/inv_aug_noref_noimg.json", prepare_fn(tokenizer))
    config = KieConfig(
        backbone_name="bros-base-uncased",
        word_embedding_name=tokenizer_name,
        head_dims=256,
        num_classes=15,
    )

    print(len(dataloader.dataset.classes))
    assert len(dataloader.dataset.classes) == config.num_classes - 1
    model = KieModel(config)
    print(model)
    for batch in dataloader:
        output = model(batch)
        ic(output)
        break
