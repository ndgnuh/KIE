from typing import *
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from pydantic import BaseModel
from torch import nn, Tensor
from transformers import AutoModel
from .utils import BatchNamespace, dataclass
from .bros import BrosModel, BrosConfig
from .graph_utils import path_graph


@torch.no_grad()
def adapt_input_bros(batch):
    # B N 4 2
    boxes = batch["boxes"] * 1.0

    # Convert to xy8
    boxes = boxes.flatten(-2).float()
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
    head_dims: int
    num_classes: int
    classes: List[str]


@dataclass
class KieOutput(BatchNamespace):
    class_logits: Tensor
    relation_logits: Tensor
    loss: Optional[Tensor] = None

    @classmethod
    def excluded(cls):
        return ["loss"]

    def __post_init__(self):
        class_probs = torch.softmax(self.class_logits, dim=-1)
        self.class_scores, self.classes = torch.max(class_probs, dim=-1)

    @property
    def relations(self):
        relation_probs = torch.softmax(self.relation_logits, dim=-1)
        relation_scores = relation_probs[..., 1] - relation_probs[..., 0]
        if self.batched:
            return torch.stack([
                self.nms(score) for score in relation_scores
            ])
        else:
            return self.nms(relation_scores)

    @torch.no_grad()
    def nms(self, scores, threshold=0):
        r, c = scores.shape
        keeps = torch.zeros_like(scores)
        scores = torch.clone(scores)
        max_score = torch.inf
        for i in range(r):
            scores[i, i] = -torch.inf

        for _ in range(10000):
            idx = torch.argmax(scores, keepdims=True)
            i, j = idx // c, idx % c
            max_score = scores[i, j]
            keeps[i, j] = 1

            scores[i, :] = -torch.inf
            scores[:, j] = -torch.inf
            scores[j, i] = -torch.inf

            if max_score < threshold:
                break
        return keeps


class ClassificationHead(nn.Sequential):
    def __init__(self, config: KieConfig):
        super().__init__()
        self.classify = nn.Linear(config.head_dims, config.num_classes)


class CPRelationTaggerHead(nn.Module):
    def __init__(self, config: KieConfig):
        super().__init__()
        self.head = nn.Linear(config.head_dims, config.head_dims)
        self.tail = nn.Linear(config.head_dims, config.head_dims)
        self.predict = nn.Conv2d(1, 2, 1)

    def forward(self, hidden):
        # b n d -> b n n d
        head = self.head(hidden)
        tail = self.tail(hidden)
        scores = torch.matmul(head, tail.transpose(-1, -2))

        pool1, _ = scores.max(dim=1, keepdim=True)
        pool2, _ = scores.max(dim=2, keepdim=True)
        scores = pool1 + pool2

        # b n n -> b 1 n nn
        scores = scores.unsqueeze(1)
        scores = self.predict(scores)
        # b d n n -> b n n d
        scores = scores.permute((0, 2, 3, 1))
        return scores


class PathRelationTaggerHead(nn.Module):
    def __init__(self, config: KieConfig):
        super().__init__()
        head_dims = config.head_dims
        self.none = nn.Parameter(torch.zeros(1, 1, head_dims))
        self.head = nn.Linear(head_dims, head_dims)
        self.tail = nn.Linear(head_dims, head_dims)

    def forward(self, hidden):
        # Expand to batch
        none = self.none.repeat([hidden.shape[0], 1, 1])
        hidden = torch.cat([none, hidden], dim=1)

        # N+1 M+1 D
        head = torch.tanh(self.head(hidden))
        tail = torch.tanh(self.tail(hidden))

        # N+1 M+1
        relations_logits = torch.matmul(head, tail.transpose(-1, -2))
        return relations_logits


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
        # Ignore nega links completely
        self.r_loss = nn.CrossEntropyLoss(
            weight=torch.tensor([0.002, 1]),
            # label_smoothing=0.1,
            # reduction='none'
        )

    def forward(self, pr_class_logits, gt_classes, pr_relation_logits, gt_relations):

        # Basic losses
        # transpose 1 -1 to put the class to 1 dim
        c_loss = self.c_loss(pr_class_logits.transpose(1, -1), gt_classes)

        # ignore padding and other's tokens
        type_mask = (gt_classes != 0) & (gt_classes != 2)
        type_mask = type_mask[:, :, None] & type_mask[:, None, :]

        # weighted loss based on what we want to do
        pr_relation_scores, pr_relations = pr_class_logits.max(dim=-1)
        r_loss = F.cross_entropy(
            pr_relation_logits[type_mask], gt_relations[type_mask])
        r_loss = r_loss + F.cross_entropy(
            pr_relation_logits[~type_mask], gt_relations[~type_mask])
        r_loss /= 2
        # masks = dict(
        #     tp=type_mask & (pr_relations == 1) & (gt_relations == 1),
        #     fp=type_mask & (pr_relations == 1) & (gt_relations == 0),
        #     fn=type_mask & (pr_relations == 0) & (gt_relations == 1),
        #     tn=type_mask & (pr_relations == 0) & (gt_relations == 0),
        # )
        # counts = {k: torch.count_nonzero(v) for k, v in masks.items()}
        # totals = pr_relations.numel()
        # weights = {k: v / totals for k, v in counts.items()}
        # weights = dict(tp=1, fp=1, fn=1, tn=1)
        # r_losses = [F.cross_entropy(pr_relation_logits[mask], gt_relations[mask]) * weights[k]
        #             for k, mask in masks.items()]
        # r_losses = [loss for loss in r_losses if not torch.isnan(loss)]
        # r_loss = F.cross_entropy(
        #     pr_relation_logits[gt_relations], positive[gt_relations])
        # r_loss = self.r_loss(pr_relation_logits.transpose(1, -1), gt_relations)
        # p = torch.count_nonzero(gt_relations)
        # n = torch.count_nonzero(~gt_relations)
        # pnc = p / (n + p)
        # ic(pnc) ~ 0.0019

        # if len(r_losses) == 0:
        #     loss = c_loss
        # else:
        loss = r_loss + c_loss

        # Penalty for sub-sequence boxes
        # pr_scores, pr_classes = pr_class_logits.max(dim=-1)
        # for b, pr_classes_ in enumerate(pr_classes):
        #     for i in pr_classes_:
        #         if i == 1:
        #             # Penalty for no incoming edge
        #             probs = pr_relation_logits[b, :, i]
        #             loss = loss + (probs[0] - probs[1]).sum()
        #         elif i > 2:
        #             # Penalty for no out going edge
        #             probs = pr_relation_logits[b, i, :]
        #             loss = loss + (probs[0] - probs[1]).sum()
        return loss


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
                class_logits, batch.classes, relation_logits, batch.adj
            )
            # loss = loss + self.loss.extra_loss(hidden, batch["relations"])
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
