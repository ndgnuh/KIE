from dataclasses import dataclass
from typing import List

import numpy as np
from .data import Sample

def mode(x):
    modals, counts = np.unique(x, return_counts=True)
    index = np.argmax(counts)
    return modals[index]


def post_process(
    tokenizer,
    texts,
    boxes,
    position_ids,
    classes,
    relations,
    relation_scores=None,
    class_scores=None,
):
    #
    # Detokenize 1d properties
    #
    node_texts = []
    node_boxes = []
    node_classes = []
    node_class_scores = []
    for node_idx in np.unique(position_ids):
        # detokenize
        (token_indices,) = np.where(position_ids == node_idx)
        node_text = tokenizer.decode(texts[token_indices])
        node_class = mode([c for c in classes if c != 0])
        # node_class_score = np.mean(
        #     [score for c, score in zip(classes, class_scores) if c != 0]
        # )

        # Append
        node_texts.append(node_text)
        node_classes.append(node_class)
        # node_class_scores.append(node_class_score)

    #
    # Detokenize links
    #
    node_edges = []
    n = len(texts)
    # for i, j, score in decode_relation(relations, relation_scores):
    for i, j in zip(*np.where(relations)):
        node_i = position_ids[i]
        node_j = position_ids[j]
        node_i != node_j and node_edges.append((node_i, node_j))

    # To dict
    node_classes = {k: (v - 1) for k, v in enumerate(node_classes) if v != 0}

    return Sample(texts=node_texts, boxes=[], links=node_edges, classes=node_classes)


def decode_relation(relations, relation_scores=None, edge_threshold=0):
    """
    Return list of edges.

    1. get the edge (i, j) with highest score, append to edge_index
    2. delete all the edge (i', j') that is not possible with this edge enabled:
      - i' = i, for this graph is path-like
      - j' = j, for this graph is path-like
      - i = j, j = i, the reverse edge
    by setting the relation score to -1 (worst case).
    3. loop until the highest score edge's score is less than the threshold
    """
    if relation_scores is None:
        relation_scores = relations

    scores = relation_scores * relations
    shape = relation_scores.shape
    edges = []
    count = 0
    for count in range(shape[0] * 2):
        print(count, edges)
        i, j = np.unravel_index(np.argmax(scores), shape)

        # won't allow self loop
        if i == j:
            relation_scores[i, j] = -1
            continue

        # score threshold
        score = relation_scores[i, j]
        if score <= edge_threshold:
            break

        # append
        edges.append((i, j, score))

        # set to zero
        relation_scores[i, :] = -1  # i outgoing
        relation_scores[:, j] = -1  # j incoming
        relation_scores[j, i] = -1  # reverse link
        relation_scores[i, j] = -1  # the current one?
    return edges
